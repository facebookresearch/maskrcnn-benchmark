import torch
from torch.nn import functional as F

import pycocotools.mask as mask_util
import numpy as np
import cv2

from maskrcnn_benchmark.modeling.matcher import Matcher
# from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.rotate_ops import rotate_iou, crop_min_area_rect, paste_rotated_roi_in_image
# from maskrcnn_benchmark.modeling.rrpn.anchor_generator import convert_rect_to_pts, get_bounding_box
from maskrcnn_benchmark.modeling.rrpn.utils import get_boxlist_rotated_rect_tensor
from maskrcnn_benchmark.modeling.roi_heads.mask_head.loss import compute_mask_iou_targets

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

def compute_rotated_proposal_gt_iou(gt_mask, proposal):
    img_h, img_w = gt_mask.shape[:2]

    xc, yc, w, h, angle = proposal
    h, w = np.round([h, w]).astype(np.int32)

    if h <= 0 or w <= 0:
        return 0.0
    img_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    proposal_mask = np.ones((h, w), dtype=np.uint8)
    proposal_mask = paste_rotated_roi_in_image(img_mask, proposal_mask, proposal)
    proposal_mask = gt_mask * proposal_mask

    full_area = np.sum(gt_mask == 1)
    box_area = np.sum(proposal_mask == 1)
    mask_iou = float(box_area) / full_area
    # rle_for_fullarea = mask_util.encode(np.asfortranarray(gt_mask))
    # rle_for_box_area = mask_util.encode(np.asfortranarray(proposal_mask))
    # full_area = mask_util.area(rle_for_fullarea).sum().astype(float)
    # box_area = mask_util.area(rle_for_box_area).sum().astype(float)
    # mask_iou = box_area / full_area

    return mask_iou


def project_masks_on_boxes(segmentation_masks, proposals, discretization_size, maskiou_on=False):
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
    mask_ratios = []

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
        gt_mask = segmentation_mask.get_mask_tensor().numpy()
        # vis_mask(gt_mask * 255, proposal.numpy())

        roi = proposal.numpy()
        if np.any(np.isnan(roi)) or np.any(np.isinf(roi)) or roi[2] <= 0 or roi[3] <= 0:
            scaled_mask = np.zeros((M, M), dtype=np.float32)
        else:
            # crop the mask from the ROI and rotate to make it axis-aligned
            cropped_mask = crop_min_area_rect(gt_mask, roi)
            scaled_mask = cv2.resize(cropped_mask.astype(np.float32), (M, M))  # bilinear by default
            scaled_mask[scaled_mask < 0.5] = 0
            scaled_mask[scaled_mask >= 0.5] = 1
        #
        mask = torch.from_numpy(scaled_mask)
        masks.append(mask)

        if maskiou_on:
            mask_ratio = compute_rotated_proposal_gt_iou(gt_mask, proposal)
            mask_ratio = torch.tensor(mask_ratio)
            mask_ratios.append(mask_ratio)

    if len(masks) == 0:
        return torch.empty(0, dtype=torch.float32, device=device), torch.empty(0, dtype=torch.float32, device=device)

    if maskiou_on:
        mask_ratios = torch.stack(mask_ratios, dim=0).to(device, dtype=torch.float32)

    return torch.stack(masks, dim=0).to(device, dtype=torch.float32), mask_ratios


class MaskRCNNLossComputation(object):
    def __init__(self, proposal_matcher, discretization_size, maskiou_on=False):
        """
        Arguments:
            proposal_matcher (Matcher)
            discretization_size (int)
        """
        self.proposal_matcher = proposal_matcher
        self.discretization_size = discretization_size
        self.maskiou_on = maskiou_on

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
        mask_ratios = []

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

            masks_per_image, mask_ratios_per_image = project_masks_on_boxes(
                segmentation_masks, positive_proposals, self.discretization_size, self.maskiou_on
            )

            labels.append(labels_per_image)
            masks.append(masks_per_image)
            mask_ratios.append(mask_ratios_per_image)

        return labels, masks, mask_ratios

    def __call__(self, proposals, mask_logits, targets):
        """
        Arguments:
            proposals (list[BoxList])
            mask_logits (Tensor)
            targets (list[BoxList])

        Return:
            mask_loss (Tensor): scalar tensor containing the loss
        """
        labels, mask_targets, mask_ratios = self.prepare_targets(proposals, targets)

        labels = cat(labels, dim=0)
        mask_targets = cat(mask_targets, dim=0)

        positive_inds = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[positive_inds]

        if self.maskiou_on:
            selected_index = torch.arange(mask_logits.shape[0], device=labels.device)
            selected_mask = mask_logits[selected_index, labels]
            mask_num, mask_h, mask_w = selected_mask.shape
            selected_mask = selected_mask.reshape(mask_num, 1, mask_h, mask_w)

        # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # accept empty tensors, so handle it separately
        if positive_inds.numel() == 0 or mask_targets.numel() == 0:
            if not self.maskiou_on:
                return mask_logits.sum() * 0
            else:
                return mask_logits.sum() * 0, selected_mask, labels, None

        pred_masks = mask_logits[positive_inds, labels_pos]
        mask_loss = F.binary_cross_entropy_with_logits(pred_masks, mask_targets)

        if not self.maskiou_on:
            return mask_loss
        else:
            device = labels.device
            mask_ratios = cat(mask_ratios, dim=0)
            maskiou_targets = compute_mask_iou_targets(pred_masks, mask_ratios, mask_targets, device)

            return mask_loss, selected_mask.sigmoid(), labels, maskiou_targets


def make_roi_mask_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    loss_evaluator = MaskRCNNLossComputation(
        matcher, cfg.MODEL.ROI_MASK_HEAD.RESOLUTION, cfg.MODEL.MASKIOU_ON
    )

    return loss_evaluator
