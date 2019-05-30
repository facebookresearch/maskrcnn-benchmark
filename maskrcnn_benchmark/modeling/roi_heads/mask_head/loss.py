# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

import pycocotools.mask as mask_util
import numpy as np

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat


def compute_proposal_gt_iou(gt_seg_mask, proposal, proposal_mask=None):
    cropped_mask = gt_seg_mask.crop(proposal) if proposal_mask is None else proposal_mask

    x1 = int(proposal[0])
    y1 = int(proposal[1])
    x2 = int(proposal[2]) + 1
    y2 = int(proposal[3]) + 1

    polygons = [p.polygons for p in gt_seg_mask.instances.polygons]  # list of list
    for poly_ in polygons:
        poly = torch.cat(poly_)  # merge all subpolygons into one polygon
        x1 = np.minimum(x1, poly[0::2].min())
        x2 = np.maximum(x2, poly[0::2].max())
        y1 = np.minimum(y1, poly[1::2].min())
        y2 = np.maximum(y2, poly[1::2].max())
    img_h = gt_seg_mask.size[1]
    img_w = gt_seg_mask.size[0]
    x1 = np.maximum(x1, 0)
    x2 = np.minimum(x2, img_w - 1)
    y1 = np.maximum(y1, 0)
    y2 = np.minimum(y2, img_h - 1)
    gt_seg_mask_for_maskratio = gt_seg_mask.crop([x1, y1, x2, y2])
    ''' 
    #type 1
    gt_img_mask = gt_seg_mask_for_maskratio.convert(mode='mask')    
    gt_img_mask_area = gt_img_mask.sum().float()
    gt_box_mask = gt_img_mask[int(proposal[1]-y1):int(proposal[3]-y1)+1, int(proposal[0]-x1):int(proposal[2]-x1)+1]
    gt_box_mask_area = gt_box_mask.sum().float()
    mask_ratio = gt_box_mask_area / gt_img_mask_area
    '''
    # type 2
    fullarea_polygons = [torch.cat(p.polygons) for p in
                         gt_seg_mask_for_maskratio.instances.polygons]  # list of list
    boxarea_polygons = [torch.cat(p.polygons) for p in cropped_mask.instances.polygons]  # list of list
    rle_for_fullarea = mask_util.frPyObjects([p.numpy() for p in fullarea_polygons], y2 - y1, x2 - x1)
    full_area = mask_util.area(rle_for_fullarea).sum().astype(float)
    rle_for_box_area = mask_util.frPyObjects([p.numpy() for p in boxarea_polygons], proposal[3] - proposal[1],
                                             proposal[2] - proposal[0])
    box_area = mask_util.area(rle_for_box_area).sum().astype(float)
    mask_iou = box_area / full_area

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
    proposals = proposals.convert("xyxy")
    assert segmentation_masks.size == proposals.size, "{}, {}".format(
        segmentation_masks, proposals
    )

    # FIXME: CPU computation bottleneck, this should be parallelized
    proposals = proposals.bbox.to(torch.device("cpu"))
    for segmentation_mask, proposal in zip(segmentation_masks, proposals):
        # crop the masks, resize them to the desired resolution and
        # then convert them to the tensor representation.
        cropped_mask = segmentation_mask.crop(proposal)
        scaled_mask = cropped_mask.resize((M, M))
        mask = scaled_mask.get_mask_tensor()
        masks.append(mask)

        if maskiou_on:
            mask_ratio = compute_proposal_gt_iou(segmentation_mask, proposal, cropped_mask)
            mask_ratio = torch.tensor(mask_ratio)
            mask_ratios.append(mask_ratio)

    if maskiou_on:
        mask_ratios = torch.stack(mask_ratios, dim=0).to(device, dtype=torch.float32)
    else:
        mask_ratios = None
    if len(masks) == 0:
        return torch.empty(0, dtype=torch.float32, device=device), torch.empty(0, dtype=torch.float32, device=device)
    return torch.stack(masks, dim=0).to(device, dtype=torch.float32), mask_ratios


def compute_mask_iou_targets(pred_masks_in, mask_ratios, mask_targets, device):
    value_eps = 1e-10 * torch.ones(mask_targets.shape[0], device=device)
    mask_ratios = torch.max(mask_ratios, value_eps)
    pred_masks = (pred_masks_in > 0).to(torch.float32)
    mask_targets_full_area = mask_targets.sum(dim=[1,2]) / mask_ratios
    mask_ovr = pred_masks * mask_targets
    mask_ovr_area = mask_ovr.sum(dim=[1,2])
    mask_union_area = pred_masks.sum(dim=[1,2]) + mask_targets_full_area - mask_ovr_area
    value_1 = torch.ones(pred_masks.shape[0], device=device)
    value_0 = torch.zeros(pred_masks.shape[0], device=device)
    mask_union_area = torch.max(mask_union_area, value_1)
    mask_ovr_area = torch.max(mask_ovr_area, value_0)
    maskiou_targets = mask_ovr_area / mask_union_area
    return maskiou_targets     


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
        match_quality_matrix = boxlist_iou(target, proposal)
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
        if mask_targets.numel() == 0:
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
