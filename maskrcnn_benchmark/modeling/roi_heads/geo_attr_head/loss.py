import torch
from torch.nn import functional as F

from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.layers import smooth_l1_loss

class GEOATTRRCNNLossComputation(object):
    def __init__(self, proposal_matcher):
        """
        Arguments:
            proposal_matcher (Matcher)
        """
        self.proposal_matcher = proposal_matcher

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # GETATTR RCNN needs "labels" and "alphas", "locations", "dimensions", "relative_dimensions"
        # "rotation_y", "alpha_conf", "alpha_oriention" fields for creating the targets
        target = target.copy_with_fields(["labels", "alphas", "locations", "dimensions", 
                                        "relative_dimensions", "rotation_y", "alpha_conf", 
                                        "alpha_oriention"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        locations_targets = []
        relative_dimensions_targets = []
        alpha_conf_targets = []
        alpha_oriention_targets = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # this can probably be removed, but is left here for clarity
            # and completeness
            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0

            # mask scores are only computed on positive samples
            positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

            locations_target = matched_targets.get_field("locations")
            locations_target = locations_target[positive_inds]

            relative_dimensions_target = matched_targets.get_field("relative_dimensions")
            relative_dimensions_target = relative_dimensions_target[positive_inds]

            alpha_conf_target = matched_targets.get_field("alpha_conf")
            alpha_conf_target = alpha_conf_target[positive_inds]

            alpha_oriention_target = matched_targets.get_field("alpha_oriention")
            alpha_oriention_target = alpha_oriention_target[positive_inds]

            # alphas_target = matched_targets.get_field("alphas")
            # alphas_target = alphas_target[positive_inds]

            # dimensions_target = matched_targets.get_field("dimensions")
            # dimensions_target = dimensions_target[positive_inds]
            
            # rotation_y_target = matched_targets.get_field("rotation_y")
            # rotation_y_target = rotation_y_target[positive_inds]

            labels.append(labels_per_image)
            locations_targets.append(locations_target)
            relative_dimensions_targets.append(relative_dimensions_target)
            alpha_conf_targets.append(alpha_conf_target)
            alpha_oriention_targets.append(alpha_oriention_target)

        return labels, locations_targets, relative_dimensions_targets, alpha_conf_targets, alpha_oriention_targets

    def __call__(self, proposals, geo_attr_logits, targets):
        """
        Arguments:
            proposals (list[BoxList])
            geo_attr_logits (list(Tensor))
            targets (list[BoxList])

        Return:
            mask_loss (Tensor): scalar tensor containing the loss
        """
        labels, locations_targets, relative_dimensions_targets, alpha_conf_targets, \
            alpha_oriention_targets = self.prepare_targets(proposals, targets)

        labels = cat(labels, dim=0)
        locations_targets = cat(locations_targets, dim=0)
        relative_dimensions_targets = cat(relative_dimensions_targets, dim=0)
        alpha_conf_targets = cat(alpha_conf_targets, dim=0)
        alpha_oriention_targets = cat(alpha_oriention_targets, dim=0)

        positive_inds = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[positive_inds]

        # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # accept empty tensors, so handle it separately
        if locations_targets.numel() == 0:
            return locations_targets.sum() * 0
        
        pred_dims, pred_ori_conf, pred_ori_consin, pred_location = geo_attr_logits

        dimension_loss = smooth_l1_loss(
            pred_dims[positive_inds, labels_pos], relative_dimensions_targets
        )

        orientation_conf_loss = F.smooth_l1_loss(
            pred_ori_conf[positive_inds, labels_pos], alpha_conf_targets
        )

        orientation_reg_loss = F.smooth_l1_loss(
            pred_ori_consin[positive_inds, labels_pos], alpha_oriention_targets
        )

        location_loss = F.smooth_l1_loss(
            pred_location[positive_inds, labels_pos], locations_targets
        )

        return {"loss_geo_attr_dim": dimension_loss,
                "loss_geo_attr_ori_conf": orientation_conf_loss,
                "loss_geo_attr_ori_reg": orientation_reg_loss,
                "loss_geo_attr_loc": location_loss}


def make_roi_geo_attr_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False
    )
    loss_evaluator = GEOATTRRCNNLossComputation(proposal_matcher=matcher)
    return loss_evaluator