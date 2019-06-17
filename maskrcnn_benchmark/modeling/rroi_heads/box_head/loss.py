import torch
from torch.nn import functional as F

# from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.rotated_box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
# from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.rotate_ops import rotate_iou
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.rrpn.loss import compute_reg_targets, compute_iou_rotate_loss
from maskrcnn_benchmark.modeling.rrpn.utils import get_boxlist_rotated_rect_tensor

from maskrcnn_benchmark.modeling.utils import cat

from .inference import REGRESSION_CN


def smooth_l1_loss(input, target, beta=1. / 9):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    return loss


class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(
        self, 
        proposal_matcher, 
        fg_bg_sampler, 
        box_coder, 
        cls_agnostic_bbox_reg=False
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

    def match_targets_to_proposals(self, proposal, target):
        masks_field = "masks"
        rrects_field = "rrects"

        # anchor_tensor = anchor.get_field(rrects_field)
        proposal_tensor = get_boxlist_rotated_rect_tensor(proposal, masks_field, rrects_field)
        target_tensor = get_boxlist_rotated_rect_tensor(target, masks_field, rrects_field)
        if not target.has_field(rrects_field):  # add rrects to gt if not already added
            target.add_field(rrects_field, target_tensor)

        match_quality_matrix = rotate_iou(target_tensor, proposal_tensor)

        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields(["labels", rrects_field])
        # target.add_field(rrects_field, target_tensor)
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        regression_targets = []
        matched_gt_ids_per_image = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # compute regression targets
            regression_targets_per_image = compute_reg_targets(
                matched_targets.get_field("rrects"), proposals_per_image.get_field("rrects"), self.box_coder
            )
            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)
            matched_gt_ids_per_image.append(matched_idxs)

        return labels, regression_targets, matched_gt_ids_per_image

    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """

        labels, regression_targets, matched_gt_ids = self.prepare_targets(proposals, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, regression_targets_per_image, m_gt_pi, proposals_per_image in zip(
            labels, regression_targets, matched_gt_ids, proposals
        ):
            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field(
                "regression_targets", regression_targets_per_image
            )
            # proposals_per_image.add_field(
            #     "matched_idxs", m_gt_pi
            # )

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

        self._proposals = proposals
        # self._targets = targets
        return proposals

    def __call__(self, class_logits, box_regression):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        device = class_logits.device

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposals

        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        regression_targets = cat(
            [proposal.get_field("regression_targets") for proposal in proposals], dim=0
        )

        labels_gt_0 = labels > 0
        sampled_pos_inds = torch.nonzero(labels_gt_0).squeeze(1)
        labels_pos = labels[sampled_pos_inds]
        total_pos = labels_pos.numel()
        total_samples = labels.numel()
        total_neg = total_samples - total_pos

        if total_pos == 0:
            return class_logits.sum() * 0, class_logits.sum() * 0

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        if self.cls_agnostic_bbox_reg:
            map_inds = torch.arange(REGRESSION_CN, REGRESSION_CN*2, device=device)
        else:
            map_inds = REGRESSION_CN * labels_pos[:, None] + torch.arange(REGRESSION_CN, device=device)

        # pos_box_regression = box_regression[sampled_pos_inds[:, None], map_inds]
        # pos_reg_targets = regression_targets[sampled_pos_inds]

        # # NEW
        # targets = self._targets
        #
        # matched_gt_ids = [p.get_field("matched_idxs") for p in proposals]
        #
        # with torch.no_grad():
        #     start_gt_idx = 0
        #     for ix, t in enumerate(targets):
        #         matched_gt_ids[ix] += start_gt_idx
        #         start_gt_idx += len(t)
        #
        #     matched_gt_ids = torch.cat(matched_gt_ids)
        #     pos_matched_gt_ids = matched_gt_ids[sampled_pos_inds]
        #
        #     pos_label_weights = torch.zeros_like(pos_matched_gt_ids, dtype=torch.float32)
        #
        #     label_idxs = [torch.nonzero(pos_matched_gt_ids == x).squeeze() for x in range(start_gt_idx)]
        #
        #     # """OLD"""
        #     label_cnts = [li.numel() for li in label_idxs]
        #     # label_weights = total_pos / label_cnts.to(dtype=torch.float32)
        #     # label_weights /= start_gt_idx  # equal class weighting
        #     for x in range(start_gt_idx):
        #         if label_cnts[x] > 0:
        #             pos_label_weights[label_idxs[x]] = total_pos / label_cnts[x] / start_gt_idx  # equal class weighting
        #
        # # # perform weighted classification loss (to prevent class imbalance i.e. too many negative)
        # num_classes = class_logits.shape[-1]
        # # with torch.no_grad():
        # #     num_classes = class_logits.shape[-1]
        # #     label_cnts = torch.stack([(labels == x).sum() for x in range(num_classes)])
        # #     label_weights = 1.0 / label_cnts.to(dtype=torch.float32)
        # #     label_weights /= num_classes   # equal class weighting

        pos_frac = self.fg_bg_sampler.positive_fraction
        classification_loss = F.cross_entropy(class_logits, labels, reduce=False)#, weight=label_weights)
        cls_weights = torch.ones_like(labels, dtype=torch.float32)
        if total_pos > 0:
            cls_weights[sampled_pos_inds] = pos_frac / total_pos #/ num_classes
        if total_neg > 0:
            cls_weights[torch.nonzero(~labels_gt_0).squeeze(1)] = (1.0 - pos_frac) / total_neg# / num_classes
        classification_loss = torch.mul(classification_loss, cls_weights).sum()

        # classification_loss = F.cross_entropy(class_logits, labels)

        # box_loss = smooth_l1_loss(
        #     pos_box_regression,
        #     pos_reg_targets,
        #     # size_average=True,
        #     beta=1.0,
        # )
        # box_loss = box_loss.mean() # (box_loss * pos_label_weights.unsqueeze(1)).sum() / total_samples
        # # box_loss = box_loss / labels.numel()

        # box_loss = smooth_l1_loss(
        #     box_regression[sampled_pos_inds[:, None], map_inds],
        #     regression_targets[sampled_pos_inds],
        #     # size_average=False,
        #     beta=1,
        # ).sum()
        # box_loss = box_loss / total_samples

        box_reg = box_regression[sampled_pos_inds[:, None], map_inds]
        box_reg_targets = regression_targets[sampled_pos_inds]

        base_anchors = torch.cat([a.get_field("rrects") for a in proposals])[sampled_pos_inds]
        pred_box = self.box_coder.decode(box_reg, base_anchors)
        gt_box = self.box_coder.decode(box_reg_targets, base_anchors)
        ious = compute_iou_rotate_loss(pred_box, gt_box)
        iou_loss = torch.where(ious <= 0, ious * 0.0, -torch.log(ious ** 2))
        box_loss = iou_loss.sum() / total_samples

        return classification_loss, box_loss


def make_roi_box_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    reg_angle_relative = cfg.MODEL.ROI_HEADS.BBOX_REG_ANGLE_RELATIVE
    box_coder = BoxCoder(weights=bbox_reg_weights, relative_angle=reg_angle_relative)

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )

    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    loss_evaluator = FastRCNNLossComputation(
        matcher, 
        fg_bg_sampler, 
        box_coder, 
        cls_agnostic_bbox_reg
    )

    return loss_evaluator
