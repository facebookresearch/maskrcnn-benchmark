# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
from torch import nn

from maskrcnn_benchmark.layers import ROIAlign

from .utils import cat
from .utils import keep_only_positive_boxes
from .utils import nonzero


class Pooler(nn.Module):
    """
    Pooler for Detection with or without FPN.
    It currently hard-code ROIAlign in the implementation,
    but that can be made more generic later on.
    Also, the requirement of passing the scales is not strictly necessary, as they
    can be inferred from the size of the feature map / size of original image,
    which is available thanks to the BoxList.
    """

    def __init__(self, output_size, scales, sampling_ratio, drop_last):
        """
        Arguments:
            output_size (list[tuple[int]] or list[int]): output size for the pooled region
            scales (list[flaot]): scales for each Pooler
            sampling_ratio (int): sampling ratio for ROIAlign
            drop_last (bool): if passed, drop the last feature map
        """
        super(Pooler, self).__init__()
        poolers = []
        for scale in scales:
            poolers.append(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio
                )
            )
        self.poolers = nn.ModuleList(poolers)
        self.drop_last = drop_last

    def forward(self, x, boxes):
        """
        Arguments:
            x (list[Tensor]): feature maps for each level
            boxes (list[list[BoxList]]): boxes to be used to perform the pooling operation.
                The first level corresponds to the feature map level, and the second
                to different input images.
        Returns:
            result (Tensor)
        """
        if self.drop_last:
            x = x[:-1]
        assert len(boxes) == len(self.poolers)
        assert len(boxes) == len(x)
        result = []
        for per_level_feature, per_level_boxes, pooler in zip(x, boxes, self.poolers):
            # TODO the scales can be inferred from the bboxes and the feature maps
            # print(per_level_boxes[0].size[::-1], per_level_feature.shape)
            ids = [
                i for i, l in enumerate(per_level_boxes) for _ in range(l.bbox.shape[0])
            ]
            concat_boxes = torch.cat([b.bbox for b in per_level_boxes], dim=0)
            ids = concat_boxes.new_tensor(ids)
            concat_boxes = torch.cat([ids[:, None], concat_boxes], dim=1)
            result.append(pooler(per_level_feature, concat_boxes))
        return cat(result, dim=0)


class MaskFPNPooler(Pooler):
    """
    This pooler is used for both training and inference.
    The behavior of the pooler changes if it's in training or inference.

    During training:
    The behavior is the same as in FPNPooler, except that we
    filter out all the non-positive boxes before passing it to
    the pooler. This saves compute and memory.

    During inference:
    It takes a set of bounding boxes (one per image), splits them
    in several feature map levels, process each level independently,
    concatenates the results from all the levels and then permute
    the results so that they are in the original order.
    """

    def __init__(
        self, output_size, scales, sampling_ratio, drop_last, roi_to_fpn_level_mapper
    ):
        super(MaskFPNPooler, self).__init__(
            output_size, scales, sampling_ratio, drop_last
        )
        self.roi_to_fpn_level_mapper = roi_to_fpn_level_mapper

    def forward(self, x, boxes):
        """
        Arguments:
            x (list[Tensor]): feature maps for each level
            boxes (list[list[BoxList]] or list[BoxList]): boxes
                to be used to perform the cropping.
                If in training mode, boxes is a list[list[BoxList]],
                where the first dimension is the feature maps,
                and the second is the image.
                In in eval mode, boxes is a list[BoxList], where
                each element in the list correspond to a different
                image.
        """

        # if it's in training format, fall back to standard
        # FPNPooler implementation
        if isinstance(boxes[0], (list, tuple)):
            # use the labels that were added by Faster R-CNN
            # subsampling to select only the positive
            # boxes -- this saves computation as usually only
            # 1 / 4 of the boxes are positive (and thus used
            # during loss computation)
            boxes = keep_only_positive_boxes(boxes)
            return super(MaskFPNPooler, self).forward(x, boxes)

        # TODO maybe factor this out in a helper class
        # This is very similar to region_proposal.py FPN
        lvl_min = self.roi_to_fpn_level_mapper.k_min
        lvl_max = self.roi_to_fpn_level_mapper.k_max
        fpn_boxes = []

        # get the permutation order to revert the fpn grouping
        box_data = cat([bbox.bbox for bbox in boxes], dim=0)
        levels = self.roi_to_fpn_level_mapper(box_data)
        rois_idx_order = [
            nonzero(levels == feat_lvl)[0] for feat_lvl in range(lvl_min, lvl_max + 1)
        ]

        # for each image, split in different fpn levels
        for img_idx, bboxes in enumerate(boxes):
            box_data = bboxes.bbox
            per_img_boxes = []
            levels = self.roi_to_fpn_level_mapper(box_data)
            for feat_lvl in range(lvl_min, lvl_max + 1):
                # TODO this can probably be optimized as has already been computed just before
                lvl_idx_per_img = nonzero(levels == feat_lvl)[0]
                bbox = bboxes[lvl_idx_per_img]
                per_img_boxes.append(bbox)
            fpn_boxes.append(per_img_boxes)

        # invert box representation to be first number of levels, and then
        # number of images
        fpn_boxes = list(zip(*fpn_boxes))

        _, rois_idx_restore = torch.sort(torch.cat(rois_idx_order, 0))

        result = super(MaskFPNPooler, self).forward(x, fpn_boxes)
        result = result[rois_idx_restore]
        return result
