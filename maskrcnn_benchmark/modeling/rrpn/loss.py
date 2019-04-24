# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
This file contains specific functions for computing losses on the RPN
file
"""

import torch
from torch.nn import functional as F

# from .utils import concat_box_prediction_layers

from ..balanced_positive_negative_sampler import BalancedPositiveNegativeSampler
# from ..utils import cat

# from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.rotate_ops import rotate_iou
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.rrpn.utils import \
    get_boxlist_rotated_rect_tensor, concat_box_prediction_layers

def smooth_l1_loss(input, target, beta=1. / 9):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    return loss

def compute_reg_targets(targets, anchors, box_coder):
    dims = targets.shape
    assert len(dims) == 2 and dims[1] == 5 and anchors.shape == dims

    reg_angles = targets[:, -1] - anchors[:, -1]
    reg_angles_sign = (reg_angles > 0).to(torch.float32)
    reg_angles_sign[reg_angles_sign == 0] = -1
    reg_angles_abs = torch.abs(reg_angles)
    gt_45 = reg_angles_abs > 45 # np.deg2rad(45) # TODO: ASSUMES ANGLES ARE ONLY -90 to 0

    # targets_copy = targets.clone()
    targets[gt_45, -1] -= reg_angles_sign[gt_45] * 90 # np.deg2rad(90)

    xd = targets[gt_45, 2:4]
    targets[gt_45, 2] = xd[:, 1]
    targets[gt_45, 3] = xd[:, 0]

    # compute regression targets
    reg_targets = box_coder.encode(
        targets, anchors
    )

    return reg_targets

class RPNLossComputation(object):
    """
    This class computes the RPN loss.
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder):
                 #generate_labels_func):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        # self.target_preparator = target_preparator
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.copied_fields = []
        # self.generate_labels_func = generate_labels_func
        self.discard_cases = ['not_visibility', 'between_thresholds']

    def match_targets_to_anchors(self, anchor, target, copied_fields=[]):
        """
        :param anchor: BoxList
        :param target: BoxList
        :param copied_fields: list
        :return:
        """
        masks_field = "masks"
        rrects_field = "rrects"

        # anchor_tensor = anchor.get_field(rrects_field)
        anchor_tensor = get_boxlist_rotated_rect_tensor(anchor, masks_field, rrects_field)
        target_tensor = get_boxlist_rotated_rect_tensor(target, masks_field, rrects_field)

        match_quality_matrix = rotate_iou(target_tensor, anchor_tensor)

        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # RPN doesn't need any fields from target
        # for creating the labels, so clear them all
        target = target.copy_with_fields(copied_fields)
        target.add_field(rrects_field, target_tensor)
        # get the targets corresponding GT for each anchor
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_idxs_clamped = matched_idxs.clamp(min=0)
        matched_targets = target[matched_idxs_clamped]
        matched_ious = match_quality_matrix[matched_idxs_clamped,
                        torch.arange(len(anchor_tensor), device=anchor_tensor.device)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        matched_targets.add_field("matched_ious", matched_ious)
        return matched_targets

    def prepare_targets(self, anchors, targets):
        labels = []
        regression_targets = []
        matched_gt_ids_per_image = []
        matched_gt_ious_per_image = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            matched_targets = self.match_targets_to_anchors(
                anchors_per_image, targets_per_image, self.copied_fields
            )

            matched_idxs = matched_targets.get_field("matched_idxs")
            labels_per_image = matched_idxs >= 0
            labels_per_image = labels_per_image.to(dtype=torch.float32)

            # Background (negative examples)
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0

            # discard anchors that go out of the boundaries of the image
            if "not_visibility" in self.discard_cases:
                labels_per_image[~anchors_per_image.get_field("visibility")] = -1

            # discard indices that are between thresholds
            if "between_thresholds" in self.discard_cases:
                inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1

            target_rrects = matched_targets.get_field("rrects")
            anchor_rrects = anchors_per_image.get_field("rrects")

            # compute regression targets
            regression_targets_per_image = compute_reg_targets(
                target_rrects, anchor_rrects, self.box_coder
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)
            matched_gt_ids_per_image.append(matched_idxs)
            matched_gt_ious_per_image.append(matched_targets.get_field("matched_ious"))

        return labels, regression_targets, matched_gt_ids_per_image, matched_gt_ious_per_image


    def __call__(self, anchors, objectness, box_regression, targets):
        """
        Arguments:
            anchors (list[BoxList])
            objectness (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor
        """

        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]

        labels, regression_targets, matched_gt_ids, \
            matched_gt_ious = self.prepare_targets(anchors, targets)

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        total_pos = sampled_pos_inds.numel()
        total_neg = sampled_neg_inds.numel()
        total_samples = total_pos + total_neg

        objectness, box_regression = concat_box_prediction_layers(objectness, box_regression)
        objectness = objectness.squeeze()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        with torch.no_grad():
            start_gt_idx = 0
            for ix, t in enumerate(targets):
                matched_gt_ids[ix] += start_gt_idx
                start_gt_idx += len(t)

            matched_gt_ids = torch.cat(matched_gt_ids)
            pos_matched_gt_ids = matched_gt_ids[sampled_pos_inds]

            label_idxs = [(pos_matched_gt_ids == x) for x in range(start_gt_idx)]
            label_cnts = torch.stack([li.sum() for li in label_idxs])
            label_weights = total_pos / label_cnts.to(dtype=torch.float32)
            label_weights /= start_gt_idx  # equal class weighting
            pos_label_weights = torch.zeros_like(pos_matched_gt_ids, dtype=torch.float32)
            for x in range(start_gt_idx):
                if label_cnts[x] > 0:
                    pos_label_weights[label_idxs[x]] = label_weights[x]

        pos_regression = box_regression[sampled_pos_inds]
        pos_regression_targets = regression_targets[sampled_pos_inds]
        # normalize_reg_targets(pos_regression_targets)
        box_loss = smooth_l1_loss(
            pos_regression,#[:, :-1],
            pos_regression_targets,#[:, :-1],
            beta=1.0 / 9,
        )
        box_loss = (box_loss * pos_label_weights.unsqueeze(1)).sum() / total_pos

        angle_loss = 0 #torch.abs(torch.sin(pos_regression[:, -1] - pos_regression_targets[:, -1])).mean()

        # balance negative and positive weights
        sampled_labels = labels[sampled_inds]
        objectness_weights = torch.ones_like(sampled_labels)
        objectness_weights[sampled_labels == 1] = 0.5 * pos_label_weights / total_pos
        objectness_weights[sampled_labels != 1] = 0.5 * 1.0 / total_neg

        criterion = torch.nn.BCELoss(reduce=False)
        entropy_loss = criterion(objectness[sampled_inds].sigmoid(), sampled_labels)
        objectness_loss = torch.mul(entropy_loss, objectness_weights).sum()

        # objectness_loss = F.binary_cross_entropy_with_logits(
        #     objectness[sampled_inds], sampled_labels, weight=objectness_weights
        # )

        return objectness_loss, box_loss, angle_loss


def make_rpn_loss_evaluator(cfg, box_coder):
    matcher = Matcher(
        cfg.MODEL.RPN.FG_IOU_THRESHOLD,
        cfg.MODEL.RPN.BG_IOU_THRESHOLD,
        allow_low_quality_matches=True,
    )

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION
    )

    loss_evaluator = RPNLossComputation(
        matcher,
        fg_bg_sampler,
        box_coder,
        # generate_rpn_labels
    )
    return loss_evaluator
