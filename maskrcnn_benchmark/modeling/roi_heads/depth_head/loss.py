# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

# from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.segmentation_mask import Polygons


def berhu_loss(input, target, beta=0.2, size_average=True):
    abs_error = torch.abs(input - target)
    max_err = torch.max(abs_error)
    min_err = torch.min(abs_error)
    if max_err - min_err > 0.5:
        c = beta * torch.max(abs_error)
        cond = abs_error <= c
        loss = torch.where(cond, abs_error, (abs_error ** 2 + c ** 2) / (2 * c))
    else:
        loss = abs_error
    if size_average:
        return loss.mean()
    return loss.sum()

def mean_var_loss(input, target, size_average=True):
    # mean_p = torch.mean(input)
    # mean_t = torch.mean(target)
    diff = target - input
    # mean_loss = torch.abs(mean_t - mean_p)
    md = torch.mean(diff)
    var_loss = (1 + torch.abs(diff - md)) ** 2 - 1
    loss = 0.5 * torch.abs(diff) + 0.5 * var_loss
    if size_average:
        return loss.mean()
    return loss.sum()


def project_masks_on_boxes(input_masks, bboxes, discretization_size, device="cpu"):
    """
    Given masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.

    Arguments:
        proposals: an instance of BoxList
    """
    masks = []
    M = discretization_size
    # assert input_masks.size == proposals.size, "{}, {}".format(
    #     input_masks, proposals
    # )
    # TODO put the proposals on the CPU, as the representation for the
    # masks is not efficient GPU-wise (possibly several small tensors for
    # representing a single instance mask)
    for in_mask, bbox in zip(input_masks, bboxes):
        # crop the masks, resize them to the desired resolution and
        # then convert them to the tensor representation,
        # instead of the list representation that was used
        cropped_mask = in_mask.crop(bbox)
        scaled_mask = cropped_mask.resize((M, M))
        if isinstance(scaled_mask, Polygons):
            scaled_mask = scaled_mask.convert(mode="mask")
            masks.append(scaled_mask.data)
        else:
            masks.append(scaled_mask.data[0])
    if len(masks) == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.stack(masks, dim=0).to(device, dtype=torch.float32)


class DepthRCNNLossComputation(object):
    def __init__(self, proposal_matcher, discretization_size):
        """
        Arguments:
            proposal_matcher (Matcher)
            discretization_size (int)
        """
        self.proposal_matcher = proposal_matcher
        self.discretization_size = discretization_size

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Mask RCNN needs "labels" and "masks "fields for creating the targets
        target = target.copy_with_fields(["labels", "depth", "masks"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        depth_masks = []
        seg_masks = []
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

            match_depth_masks = matched_targets.get_field("depth")
            match_depth_masks = match_depth_masks[positive_inds]

            match_seg_masks = matched_targets.get_field("masks")
            match_seg_masks = match_seg_masks[positive_inds]

            positive_proposals = proposals_per_image[positive_inds]
            device = positive_proposals.bbox.device
            positive_proposals = positive_proposals.convert("xyxy")
            bboxes = positive_proposals.bbox.to(torch.device("cpu"))

            depth_masks_per_image = project_masks_on_boxes(
                match_depth_masks, bboxes, self.discretization_size, device
            )
            seg_masks_per_image = project_masks_on_boxes(
                match_seg_masks, bboxes, self.discretization_size, device
            )

            labels.append(labels_per_image)
            depth_masks.append(depth_masks_per_image)
            seg_masks.append(seg_masks_per_image)

        return labels, depth_masks, seg_masks

    def __call__(self, proposals, depth_pred, targets):
        """
        Arguments:
            proposals (list[BoxList])
            depth_pred (Tensor)
            targets (list[BoxList])

        Return:
            depth_loss (Tensor): scalar tensor containing the loss
        """
        labels, depth_targets, seg_masks = self.prepare_targets(proposals, targets)

        labels = cat(labels, dim=0)
        depth_targets = cat(depth_targets, dim=0).squeeze()
        seg_masks = cat(seg_masks, dim=0).squeeze()

        positive_inds = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[positive_inds]

        # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # accept empty tensors, so handle it separately
        if depth_targets.numel() == 0:
            return depth_pred.sum() * 0

        # depth_targets = torch.log(depth_targets)  # regress to the log of depth

        dp = depth_pred[positive_inds, labels_pos]

        depth_loss = 0.0
        N = dp.size(0)
        for i in range(N):
            seg = seg_masks[i]
            pred = dp[i][seg==1]
            tar = depth_targets[i][seg==1]
            # loss = berhu_loss(pred, tar, beta=0.2, size_average=True)
            loss = mean_var_loss(pred, tar, size_average=True)
            depth_loss += loss
        depth_loss /= N

        # depth_loss = torch.abs(dp[seg_masks==1] - depth_targets[seg_masks==1]).mean() # smooth_l1_loss(dp, depth_targets)
        return depth_loss


def make_roi_depth_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_DEPTH_HEAD.FG_IOU_THRESHOLD,
        0,
        allow_low_quality_matches=False,
    )

    resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION / 2 * cfg.MODEL.ROI_DEPTH_HEAD.UPSAMPLE_FACTOR
    loss_evaluator = DepthRCNNLossComputation(
        matcher, resolution
    )

    return loss_evaluator
