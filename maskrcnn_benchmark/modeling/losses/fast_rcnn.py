# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

import torch
from torch.nn import functional as F
from ..utils import cat
from ..utils import cat_bbox
from ..utils import nonzero
from ..utils import smooth_l1_loss
from .matcher import Matcher
from .target_preparator import TargetPreparator


class FastRCNNTargetPreparator(TargetPreparator):
    """
    This class returns labels and regression targets for Fast R-CNN
    """

    def index_target(self, target, index):
        target = target.copy_with_fields("labels")
        return target[index]

    def prepare_labels(self, matched_targets_per_image, anchors_per_image):
        matched_idxs = matched_targets_per_image.get_field("matched_idxs")
        labels_per_image = matched_targets_per_image.get_field("labels")
        labels_per_image = labels_per_image.to(dtype=torch.int64)

        # Label background (below the low threshold)
        bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
        labels_per_image[bg_inds] = 0

        # Label ignore proposals (between low and high thresholds)
        ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
        labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler
        return labels_per_image


class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(self, target_preparator, fg_bg_sampler):
        """
        Arguments:
            target_preparator: an instance of TargetPreparator
            fg_bg_sampler: an instance of BalancedPositiveNegativeSampler
        """
        self.target_preparator = target_preparator
        self.fg_bg_sampler = fg_bg_sampler

    def subsample(self, anchors, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled anchors.
        Note: this function keeps a state.

        Arguments:
            anchors (list of list of BoxList)
            targets (list of BoxList)
        """

        labels, regression_targets = self.target_preparator(anchors, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        # flip anchors to be images -> feature map levels
        if anchors:
            device = anchors[0][0].bbox.device
        anchors = list(zip(*anchors))

        levels = [
            torch.tensor(
                [i for i, n in enumerate(anchor) for _ in range(n.bbox.shape[0])], device=device
            )
            for anchor in anchors
        ]
        num_levels = len(anchors[0])
        num_images = len(anchors)
        # concatenate all anchors for the same image
        anchors = [cat_bbox(anchors_per_image) for anchors_per_image in anchors]

        # add corresponding label information to the bounding boxes
        # this can be used with `keep_only_positive_boxes` in order to
        # restrict the set of boxes to be used during other steps (Mask R-CNN
        # for example)
        for labels_per_image, anchors_per_image in zip(labels, anchors):
            anchors_per_image.add_field("labels", labels_per_image)

        sampled_inds = []
        sampled_image_levels = []
        # distributed sampled anchors, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = pos_inds_img | neg_inds_img
            anchors_per_image = anchors[img_idx][img_sampled_inds]
            sampled_levels = levels[img_idx].index_select(0,nonzero(img_sampled_inds)[0])
            # TODO replace with bincount because indices in the same level
            # are packed together
            anchors_per_level_per_image = []
            sampled_image_level_temp = []
            for level in range(num_levels):
                level_idx = nonzero(sampled_levels == level)[0]
                anchors_per_level_per_image.append(anchors_per_image[level_idx])
                sampled_image_level_temp.append(torch.full_like(level_idx, img_idx))
            anchors[img_idx] = anchors_per_level_per_image
            sampled_inds.append(img_sampled_inds)
            sampled_image_levels.append(sampled_image_level_temp)
        # flip back to original format feature map level -> images
        anchors = list(zip(*anchors))

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        self._labels = labels
        self._regression_targets = regression_targets
        self._sampled_pos_inds = sampled_pos_inds
        self._sampled_neg_inds = sampled_neg_inds
        self._sampled_inds = sampled_inds

        # find permutation that brings the concatenated representation in the order
        # that first joins the images for the same level, and then concatenates the
        # levels into the representation obtained by concatenating first the feature maps
        # and then the images
        sampled_image_levels = list(zip(*sampled_image_levels))
        sampled_image_levels = cat([cat(l, dim=0) for l in sampled_image_levels], dim=0)
        permute_inds = cat(
            [
                nonzero(sampled_image_levels == img_idx)[0]
                for img_idx in range(num_images)
            ],
            dim=0,
        )

        self._permute_inds = permute_inds

        return anchors

    def __call__(self, class_logits, box_regression):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list of tensor)
            box_regression (list of tensor)
        """

        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        device = class_logits.device

        if not hasattr(self, "_labels"):
            raise RuntimeError("subsample needs to be called before")
        labels = self._labels
        regression_targets = self._regression_targets
        sampled_pos_inds = torch.cat(self._sampled_pos_inds, dim=0)
        sampled_neg_inds = torch.cat(self._sampled_neg_inds, dim=0)
        sampled_inds = torch.cat(self._sampled_inds, dim=0)

        permute_inds = self._permute_inds
        assert len(permute_inds) == len(class_logits)

        class_logits = class_logits[permute_inds]
        box_regression = box_regression[permute_inds]

        # delete cached elements
        for attr in [
            "_labels",
            "_regression_targets",
            "_sampled_pos_inds",
            "_sampled_neg_inds",
            "_sampled_inds",
            "_permute_inds",
        ]:
            delattr(self, attr)

        # get indices of the positive examples in the subsampled space
        markers = torch.arange(sampled_inds.sum(), device=device)
        marked_sampled_inds = torch.zeros(
            sampled_inds.shape[0], dtype=torch.int64, device=device
        )
        marked_sampled_inds[sampled_inds] = markers
        sampled_pos_inds_subset = marked_sampled_inds[sampled_pos_inds]

        sampled_pos_inds = nonzero(sampled_pos_inds)[0]
        sampled_neg_inds = nonzero(sampled_neg_inds)[0]
        sampled_inds = nonzero(sampled_inds)[0]

        classification_loss = F.cross_entropy(class_logits, labels[sampled_inds])

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        labels_pos = labels[sampled_pos_inds]
        map_inds = 4 * labels_pos[:, None] + torch.tensor([0, 1, 2, 3], device=device)

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds_subset[:, None], map_inds],
            regression_targets[sampled_pos_inds],
            size_average=False,
            beta=1,
        ) / (sampled_inds.numel())

        return classification_loss, box_loss


# FIXME merge this with FastRCNNLossComputation
class FastRCNNOHEMLossComputation(object):
    """
    This class computes the Fast R-CNN loss
    In an OHEM manner.
    """

    def __init__(self, target_preparator, fg_bg_sampler):
        self.target_preparator = target_preparator
        self.fg_bg_sampler = fg_bg_sampler

    def __call__(self, anchors, class_logits, box_regression, targets):
        assert len(anchors) == 1, "only single feature map supported"
        assert len(class_logits) == 1, "only single feature map supported"
        anchors = anchors[0]
        class_logits = class_logits[0]
        box_regression = box_regression[0]

        # TODO test if this works for multi-feature maps
        # assert len(anchors) == len(class_logits)
        # class_logits = cat(class_logits, dim=0)
        # box_regression = cat(box_regression, dim=0)

        device = class_logits.device

        labels, regression_targets = self.target_preparator(anchors, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = nonzero(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = nonzero(torch.cat(sampled_neg_inds, dim=0))[0]
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        classification_loss = F.cross_entropy(
            class_logits[sampled_inds], labels[sampled_inds]
        )

        # FIXME workaround because can't unsqueeze empty tensor in PyTorch
        # when there are no positive labels
        if len(sampled_pos_inds) == 0:
            box_loss = torch.tensor(0., device=device, requires_grad=True)
            return classification_loss, box_loss

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        labels_pos = labels[sampled_pos_inds]
        map_inds = 4 * labels_pos[:, None] + torch.tensor([0, 1, 2, 3], device=device)

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds[:, None], map_inds],
            regression_targets[sampled_pos_inds],
            size_average=False,
            beta=1,
        ) / (sampled_inds.numel())

        return classification_loss, box_loss
