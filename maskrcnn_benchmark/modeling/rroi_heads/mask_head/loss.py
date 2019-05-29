import torch
from torch.nn import functional as F

import numpy as np
import cv2

from maskrcnn_benchmark.modeling.matcher import Matcher
# from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.rotate_ops import rotate_iou, crop_min_area_rect
from maskrcnn_benchmark.modeling.rrpn.utils import get_boxlist_rotated_rect_tensor

from maskrcnn_benchmark.modeling.utils import cat


# def vis_mask(mask, roi):
#     import cv2
#     from maskrcnn_benchmark.modeling.rrpn.anchor_generator import draw_anchors
#
#     cropped = crop_min_area_rect(mask, roi)
#     mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
#     mask_color = draw_anchors(mask_color, [roi], [[0,0,255]])
#     cv2.imshow("mask", mask)
#     cv2.imshow("mask", mask_color)
#     cv2.imshow("cropped", cropped)
#     cv2.waitKey(0)


def project_masks_on_boxes(segmentation_masks, proposals, discretization_size):
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.

    Arguments:
        segmentation_masks: an instance of SegmentationMask
        proposals: an instance of BoxList
    """
    masks = []
    M = discretization_size
    device = proposals.bbox.device
    assert segmentation_masks.size == proposals.size, "{}, {}".format(
        segmentation_masks, proposals
    )

    # FIXME: CPU computation bottleneck, this should be parallelized
    # proposals = proposals.bbox.to(torch.device("cpu"))
    proposals = proposals.get_field("rrects").to(torch.device("cpu"))
    for segmentation_mask, proposal in zip(segmentation_masks, proposals):
        # crop the masks, resize them to the desired resolution and
        # then convert them to the tensor representation.
        bin_mask = segmentation_mask.get_mask_tensor().numpy()
        # vis_mask(bin_mask * 255, proposal.numpy())

        roi = proposal.numpy()
        if np.any(np.isnan(roi)) or np.any(np.isinf(roi)) or roi[2] <= 0 or roi[3] <= 0:
            scaled_mask = np.zeros((M, M), dtype=np.float32)
        else:
            cropped_mask = crop_min_area_rect(bin_mask, roi)
            scaled_mask = cv2.resize(cropped_mask.astype(np.float32), (M, M))  # bilinear by default

            scaled_mask[scaled_mask < 0.5] = 0
            scaled_mask[scaled_mask >= 0.5] = 1
        mask = torch.from_numpy(scaled_mask)
        masks.append(mask)
    if len(masks) == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.stack(masks, dim=0).to(device, dtype=torch.float32)


class MaskRCNNLossComputation(object):
    def __init__(self, proposal_matcher, discretization_size):
        """
        Arguments:
            proposal_matcher (Matcher)
            discretization_size (int)
        """
        self.proposal_matcher = proposal_matcher
        self.discretization_size = discretization_size

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

        # Mask RCNN needs "labels" and "masks "fields for creating the targets
        target = target.copy_with_fields(["labels", "masks"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets


    def prepare_targets(self, proposals, targets):
        labels = []
        masks = []
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

            segmentation_masks = matched_targets.get_field("masks")
            segmentation_masks = segmentation_masks[positive_inds]

            positive_proposals = proposals_per_image[positive_inds]

            masks_per_image = project_masks_on_boxes(
                segmentation_masks, positive_proposals, self.discretization_size
            )

            labels.append(labels_per_image)
            masks.append(masks_per_image)

        return labels, masks

    def __call__(self, proposals, mask_logits, targets, cls_logits=None):
        """
        Arguments:
            proposals (list[BoxList])
            mask_logits (Tensor)
            targets (list[BoxList])

        Return:
            mask_loss (Tensor): scalar tensor containing the loss
        """
        labels, mask_targets = self.prepare_targets(proposals, targets)

        labels = cat(labels, dim=0)
        mask_targets = cat(mask_targets, dim=0)

        positive_inds = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[positive_inds]

        # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # accept empty tensors, so handle it separately
        if positive_inds.numel() == 0 or mask_targets.numel() == 0:
            return mask_logits.sum() * 0, mask_logits.sum() * 0

        mask_loss = F.binary_cross_entropy_with_logits(
            mask_logits[positive_inds, labels_pos], mask_targets
        )

        # classification loss
        cls_loss = None
        if cls_logits is not None:
            cls_loss = F.binary_cross_entropy_with_logits(
                cls_logits.squeeze(1), (labels > 0).to(dtype=torch.float32)
            )
        return mask_loss, cls_loss


def make_roi_mask_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    loss_evaluator = MaskRCNNLossComputation(
        matcher, cfg.MODEL.ROI_MASK_HEAD.RESOLUTION
    )

    return loss_evaluator
