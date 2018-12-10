# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat


def project_masks_on_boxes(vertex_masks, proposals, discretization_size):
    """
    Given vertex masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.

    Arguments:
        vertex_masks: an instance of SegmentationMask
        proposals: an instance of BoxList
    """
    masks = []
    M = discretization_size
    device = proposals.bbox.device
    proposals = proposals.convert("xyxy")
    assert vertex_masks.size == proposals.size, "{}, {}".format(
        vertex_masks, proposals
    )
    # TODO put the proposals on the CPU, as the representation for the
    # masks is not efficient GPU-wise (possibly several small tensors for
    # representing a single instance mask)
    proposals = proposals.bbox.to(torch.device("cpu"))
    for vertex_mask, proposal in zip(vertex_masks, proposals):
        # crop the masks, resize them to the desired resolution and
        # then convert them to the tensor representation,
        # instead of the list representation that was used
        cropped_mask = vertex_mask.crop(proposal)
        scaled_mask = cropped_mask.resize((M, M))
        # mask = scaled_mask.convert(mode="mask")
        masks.append(scaled_mask.data[0])
    if len(masks) == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.stack(masks, dim=0).to(device, dtype=torch.float32)


def smooth_l1_loss_vertex(vertex_pred, vertex_targets, vertex_weights=None, sigma=1.0):

    sigma_2 = sigma ** 2
    diff = vertex_pred - vertex_targets
    if vertex_weights is not None and vertex_weights != 1:
        diff = torch.mul(vertex_weights, diff)
    abs_diff = torch.abs(diff)
    smoothL1_sign = (abs_diff < 1. / sigma_2).clone().float()
    smoothL1_sign.detach_()
    # smoothL1_sign = smoothL1_sign.float()
    # smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_diff, 1. / sigma_2)))
    loss = torch.pow(diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
            + (abs_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    if vertex_weights is not None and vertex_weights != 1:
        loss = torch.div( torch.sum(loss), torch.sum(vertex_weights) + 1e-10 )
    return loss.mean()


class PoseRCNNLossComputation(object):
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
        target = target.copy_with_fields(["labels", "vertex"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        vert_masks = []
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

            vertex_masks = matched_targets.get_field("vertex")
            vertex_masks = vertex_masks[positive_inds]

            positive_proposals = proposals_per_image[positive_inds]

            vert_masks_per_image = project_masks_on_boxes(
                vertex_masks, positive_proposals, self.discretization_size
            )

            labels.append(labels_per_image)
            vert_masks.append(vert_masks_per_image)

        return labels, vert_masks

    def __call__(self, proposals, vertex_pred, targets):
        """
        Arguments:
            proposals (list[BoxList])
            vertex_pred (Tensor)
            targets (list[BoxList])

        Return:
            mask_loss (Tensor): scalar tensor containing the loss
        """
        labels, vertex_targets = self.prepare_targets(proposals, targets)

        labels = cat(labels, dim=0)
        vertex_targets = cat(vertex_targets, dim=0)

        positive_inds = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[positive_inds]

        # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # accept empty tensors, so handle it separately
        if vertex_targets.numel() == 0:
            return vertex_pred.sum() * 0

        vp_size = vertex_pred.shape
        N,C,H,W = vp_size
        vp = vertex_pred.view(N,-1,3,H,W)
        vp = vp[positive_inds, labels_pos]
        # mask_loss = F.binary_cross_entropy_with_logits(
        #     vertex_pred[positive_inds, labels_pos], mask_targets
        # )
        # for pos_ind, label in zip(positive_inds, labels_pos):
        #     vertex_pred.append(vertex_pred[pos_ind, label*3 : label*3+3])
        # vertex_pred = torch.stack(masks, dim=0).to(device)
        # vertex_loss = smooth_l1_loss_vertex(vp, vertex_targets) # TODO: add vertex_weights
        vertex_loss = smooth_l1_loss(vp, vertex_targets)
        return vertex_loss


def make_roi_vertex_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    loss_evaluator = PoseRCNNLossComputation(
        matcher, cfg.MODEL.ROI_VERTEX_HEAD.RESOLUTION
    )

    return loss_evaluator
