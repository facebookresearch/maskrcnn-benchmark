# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
This file contains specific functions for computing losses on the RPN
file
"""

import torch
from torch.nn import functional as F

from ..utils import cat
from ..utils import nonzero
from ..utils import smooth_l1_loss
from .matcher import Matcher
from .target_preparator import TargetPreparator


class RPNTargetPreparator(TargetPreparator):
    """
    This class returns labels and regression targets for the RPN
    """

    def index_target(self, target, index):
        # RPN doesn't need any fields from target
        # for creating the labels, so clear them all
        target = target.copy_with_fields([])
        return target[index]

    def prepare_labels(self, matched_targets_per_image, anchors_per_image):
        """
        Arguments:
            matched_targets_per_image (BoxList): a BoxList with the 'matched_idx' field set,
                containing the ground-truth targets aligned to the anchors,
                i.e., it contains the same number of elements as the number of anchors,
                and contains de best-matching ground-truth target element. This is
                returned by match_targets_to_anchors
            anchors_per_image (BoxList object)
        """
        matched_idxs = matched_targets_per_image.get_field("matched_idxs")
        labels_per_image = matched_idxs >= 0
        labels_per_image = labels_per_image.to(dtype=torch.float32)
        # discard anchors that go out of the boundaries of the image
        labels_per_image[~anchors_per_image.get_field("visibility")] = -1

        # discard indices that are between thresholds
        inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
        labels_per_image[inds_to_discard] = -1
        return labels_per_image


class RPNLossComputation(object):
    """
    This class computes the RPN loss.
    """

    def __init__(self, target_preparator, fg_bg_sampler):
        """
        Arguments:
            target_preparator (an instance of TargetPreparator)
            fb_bg_sampler (an instance of BalancedPositiveNegativeSampler)
        """
        self.target_preparator = target_preparator
        self.fg_bg_sampler = fg_bg_sampler

    def __call__(self, anchors, objectness, box_regression, targets):
        """
        Arguments:
            anchors (list of list of BoxList)
            objectness (list of tensor)
            box_regression (list of tensor)
            targets (list of BoxList)
        """
        assert len(anchors) == len(objectness)
        labels, regression_targets = self.target_preparator(anchors, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = nonzero(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = nonzero(torch.cat(sampled_neg_inds, dim=0))[0]

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness_flattened = []
        box_regression_flattened = []
        # for each feature level, permute the outputs to make them be in the
        # same format as the labels. Note that the labels are computed for
        # all feature levels concatenated, so we keep the same representation
        # for the objectness and the box_regression
        for objectness_per_level, box_regression_per_level in zip(
            objectness, box_regression
        ):
            N, A, H, W = objectness_per_level.shape
            objectness_per_level = objectness_per_level.permute(0, 2, 3, 1).reshape(
                N, -1
            )
            box_regression_per_level = box_regression_per_level.view(N, -1, 4, H, W)
            box_regression_per_level = box_regression_per_level.permute(0, 3, 4, 1, 2)
            box_regression_per_level = box_regression_per_level.reshape(N, -1, 4)
            objectness_flattened.append(objectness_per_level)
            box_regression_flattened.append(box_regression_per_level)
        # concatenate on the first dimension (representing the feature levels), to
        # take into account the way the labels were generated (with all feature maps
        # being concatenated as well)
        objectness = cat(objectness_flattened, dim=1).reshape(-1)
        box_regression = cat(box_regression_flattened, dim=1).reshape(-1, 4)

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1.0 / 9,
            size_average=False,
        ) / (sampled_inds.numel())

        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )

        return objectness_loss, box_loss
