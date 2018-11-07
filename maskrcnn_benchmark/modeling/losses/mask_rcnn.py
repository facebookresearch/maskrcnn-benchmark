# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

import torch
from torch.nn import functional as F
from maskrcnn_benchmark import _C
from ..utils import cat
from ..utils import cat_bbox
from ..utils import keep_only_positive_boxes
from ..utils import nonzero
from ..utils import split_with_sizes
from .matcher import Matcher
import itertools
from .target_preparator import TargetPreparator


class MaskTargetPreparator(TargetPreparator):
    """
    This class aligns the ground-truth targets to the anchors that are
    passed to the image.
    """

    def __init__(self, proposal_matcher, discretization_size):
        super(MaskTargetPreparator, self).__init__(proposal_matcher, None)
        self.discretization_size = discretization_size

    def index_target(self, target, index):
        """
        This function is used to index the targets, possibly only propagating a few
        fields of target (instead of them all). In this case, we only propagate
        labels and masks.

        Arguments:
            target (BoxList): an arbitrary bbox object, containing many possible fields
            index (Tensor): the indices to select.
        """

        target = target.copy_with_fields(["labels", "masks"])
        return target[index]

    def prepare_labels(self, matched_idxs_per_image, anchors_per_image, targets_per_image):
        """
        Arguments:
            matched_targets_per_image (BoxList): a BoxList with the 'matched_idx' field set,
                containing the ground-truth targets aligned to the anchors,
                i.e., it contains the same number of elements as the number of anchors,
                and contains de best-matching ground-truth target element. This is
                returned by match_targets_to_anchors
            anchors_per_image (a BoxList object)

        This method should return a single tensor, containing the labels
        for each element in the anchors
        """
        clamped_idxs=matched_idxs_per_image.clamp(min=0)
        matched_idxs = matched_idxs_per_image
        labels_per_image =targets_per_image.get_field("labels").index_select(0,clamped_idxs)
        labels_per_image = labels_per_image.to(dtype=torch.int64)

        # this can probably be removed, but is left here for clarity
        # and completeness
        neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
        labels_per_image[neg_inds] = 0

        # mask scores are only computed on positive samples
        positive_inds = nonzero(labels_per_image > 0)[0]
        positive_anchors = anchors_per_image[positive_inds]

        masks_per_image = self.project(targets_per_image, positive_anchors, clamped_idxs.index_select(0,positive_inds))
        return masks_per_image, labels_per_image

    def project(self, targets, positive_anchors, positive_inds):
        """
        Given segmentation masks and the bounding boxes corresponding
        to the location of the masks in the image, this function
        crops and resizes the masks in the position defined by the
        boxes. This prepares the masks for them to be fed to the
        loss computation as the targets.

        Arguments:
            segmentation_masks: an instance of SegmentationMask
            positive_anchors: an instance of BoxList
        """
        M = self.discretization_size
        device = positive_anchors.bbox.device
        positive_anchors = positive_anchors.convert("xyxy")
        #a list of list of list of polygon coordinates
        polygons_list=[]
        positive_inds=positive_inds.tolist()
        for idx in positive_inds:
            poly_obj = targets.get_field("masks").polygons[idx]
            polygons_per_instance=[]
            for poly in poly_obj.polygons:
                polygons_per_instance.append(poly)
            polygons_list.append(polygons_per_instance)
        dense_coordinate_vec=torch.cat(list(itertools.chain(*polygons_list))).double()
        if len(polygons_list)>0:
            masks = _C.generate_mask_targets(dense_coordinate_vec, polygons_list, positive_anchors.bbox, M)
        if len(polygons_list) == 0:
            return torch.empty(0, dtype=torch.float32, device=device)
        return masks

class MaskRCNNLossComputation(object):
    def __init__(self, target_preparator, subsample_only_positive_boxes=False):
        """
        If subsample_only_positive_boxes is False, all the boxes from the RPN
        that were passed to the detection branch will be used for mask loss
        computation. This is wasteful, as only the positive boxes are used
        for mask loss (which corresponds to 25% of a batch).
        If subsample_only_positive_boxes is True, then only the positive
        boxes are selected, but this only works with FPN-like architectures.
        """
        self.target_preparator = target_preparator
        self.subsample_only_positive_boxes = subsample_only_positive_boxes

    def prepare_targets(self, anchors, targets):
        """
        This reimplents parts of the functionality of TargetPreparator.__call__
        The reason being that we don't need bbox regression targets for
        masks, so I decided to specialize it here instead of modifying
        TargetPreparator. It might be worth considering modifying this once
        I implement keypoints
        """
        # flip anchor representation to be first images, and then feature maps
        anchors = list(zip(*anchors))
        anchors = [cat_bbox(anchor) for anchor in anchors]

        target_preparator = self.target_preparator
        # TODO assert / resize anchors to have the same .size as targets?
        matched_idxs = target_preparator.match_targets_to_anchors(anchors, targets,mask=True)
        labels = []
        masks = []
        for matched_idxs_per_image, anchors_per_image, targets_per_image in zip(
            matched_idxs, anchors, targets
        ):
            masks_per_image, labels_per_image = target_preparator.prepare_labels(
                matched_idxs_per_image, anchors_per_image, targets_per_image
            )
            labels.append(labels_per_image)
            masks.append(masks_per_image)
        return labels, masks

    def get_permutation_inds(self, anchors):
        """
        anchors is in features - images order get the permutation to make it in
        image - features order

        Arguments:
            anchors (list[list[BoxList]]): first level corresponds to the feature maps,
                and the second to the images.

        Returns:
            result (Tensor): indices that allow to convert from the feature map-first
                representation to the image-first representation
        """
        num_images = len(anchors[0])
        # flatten anchors into a single list
        flattened = [f for l in anchors for f in l]
        sizes = [i.bbox.shape[0] for i in flattened]
        device = anchors[0][0].bbox.device
        # strategy: start from the identity permutation which has the final size
        # split it according to the sizes for each feature map / image, group the
        # indices according to a list of features of list of images, invert the
        # representation to be images -> features, and then concatenate it all
        inds = torch.arange(sum(sizes), device=device)
        # can't use torch.split because of a bug with 0 in sizes
        inds = split_with_sizes(inds, sizes)
        # grouped correspond to the linear indices split in
        # features first, and then images
        grouped = [inds[i : i + num_images] for i in range(0, len(inds), num_images)]
        # convert to images first, then features by flipping the representation
        flip = list(zip(*grouped))
        # flatten the list of lists into a single list of tensors
        flip = [f for l in flip for f in l]
        return torch.cat(flip, dim=0)

    def __call__(self, anchors, mask_logits, targets):
        """
        Arguments:
            anchors (list[list[BoxList]))
            mask_logits (Tensor)
            targets (list[BoxList])

        Return:
            mask_loss (Tensor): scalar tensor containing the loss
        """
        if self.subsample_only_positive_boxes:
            anchors = keep_only_positive_boxes(anchors)
        labels, mask_targets = self.prepare_targets(anchors, targets)

        # convert from feature map-first representation to
        # image-first representation
        permutation_inds = self.get_permutation_inds(anchors)
        mask_logits = mask_logits.index_select(0,permutation_inds)

        labels = cat(labels, dim=0)
        mask_targets = cat(mask_targets, dim=0)

        positive_inds = nonzero(labels > 0)[0]
        labels_pos = labels.index_select(0,positive_inds)

        # torch.mean (in binary_cross_entropy_with_logits) does'nt
        # accept empty tensors, so handle it sepaartely
        if mask_targets.numel() == 0:
            return mask_logits.sum() * 0

        mask_loss = F.binary_cross_entropy_with_logits(
            mask_logits[positive_inds, labels_pos], mask_targets
        )
        return mask_loss
