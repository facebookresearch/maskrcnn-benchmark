# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

import torch

from maskrcnn_benchmark.layers import nms as box_nms
from maskrcnn_benchmark.structures.bounding_box import BoxList

from ..box_coder import BoxCoder
from ..box_ops import boxes_area
from ..utils import cat
from ..utils import nonzero

from maskrcnn_benchmark import _C as C

# TODO add option for different params in train / test
class RPNBoxSelector(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RPN boxes, before feeding the proposals
    to the heads
    """

    def __init__(
        self,
        pre_nms_top_n,
        post_nms_top_n,
        nms_thresh,
        min_size,
        box_coder=BoxCoder(weights=(1., 1., 1., 1.)),
    ):
        """
        Arguments:
            pre_nms_top_n (int)
            post_nms_top_n (int)
            nms_thresh (float)
            min_size (int)
            box_coder (BoxCoder)
        """
        super(RPNBoxSelector, self).__init__()
        # TODO ATTENTION, as those numbers are for single-image in Detectron, and here it's for the batch
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size

        self.box_coder = box_coder

    """
    Arguments:
        proposals: tensor (float) of size N, 4
        indices: tensor (int) of size N
        fields: map (string -> Tensor)
        targets: list of BoxList
    """
    def add_gt_proposals(self, proposals, indices, fields, targets):
        # Get the device we're operating on
        device = proposals.device

        gt_boxes = [BoxList(target.bbox, target.size, target.mode) for target in targets]

        # later cat of bbox requires all fields to be present for all bbox
        # so we need to add a dummy for objectness that's missing
        for gt_box in gt_boxes:
            gt_box.add_field("objectness",
                             torch.ones(gt_box.bbox.numel(),
                             device=device))

        # Need to append our fields to the flattened arrays
        if fields is not None:
            for field in fields.keys():
                fields[field] = torch.cat(
                        [fields[field]] + [b.get_field(field) for b in gt_boxes], dim=0
                )

        # turn boxes into a tensor
        proposals = torch.cat([proposals] + [b.bbox for b in gt_boxes], dim=0)

        if indices is not None:
            extra_indices = [
                    torch.full((t.bbox.shape[0],), idx, device=device) for idx, t in enumerate(targets)
            ]
            indices = torch.cat([indices] + extra_indices, dim=0)

        return proposals, indices, fields


    def forward_for_single_feature_map(self, anchors, objectness, box_regression):
        """
        Arguments:
            anchors: list of BoxList
            objectness: tensor of size N, A, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        device = objectness.device
        N, A, H, W = objectness.shape

        num_anchors = A * H * W
        objectness = objectness.reshape(N, -1) # Now [N, AHW]
        objectness = objectness.sigmoid()

        pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)
        objectness, topk_idx = objectness.topk(pre_nms_top_n, dim=1, sorted=True)

        use_fast_cuda_path = objectness.is_cuda
        if use_fast_cuda_path:
            # New code
            batch_idx = torch.arange(N, device=device)[:, None]

            # Get all image shapes, and cat them together
            image_shapes = [box.size[::-1] for box in anchors]
            image_shapes_cat = torch.cat([torch.tensor(box.size[::-1], device=objectness.device).float() for box in anchors])

            # Get a single tensor for all anchors
            concat_anchors = torch.cat([a.bbox for a in anchors], dim=0)

            # Note: Take all anchors, we'll index accordingly inside the kernel
            # only take the anchors corresponding to the topk boxes
            concat_anchors = concat_anchors.reshape(N, -1, 4) # [batch_idx, topk_idx]

            # Return pre-nms boxes, associated scores and keep flag
            # Encompasses:
            # 1. Box decode
            # 2. Box clipping
            # 3. Box filtering
            # At the end we need to keep only the proposals & scores flagged
            # Note: topk_idx, objectness are sorted => proposals, objectness, keep are also
            # sorted -- this is important later
            proposals, objectness, keep = C.GeneratePreNMSUprightBoxes(
                                    N,
                                    A,
                                    H,
                                    W,
                                    topk_idx,
                                    objectness.float(),    # Need to cast these as kernel doesn't support fp16
                                    box_regression.float(),
                                    concat_anchors,
                                    image_shapes_cat,
                                    pre_nms_top_n,
                                    0, # feature_stride
                                    self.min_size,
                                    self.box_coder.bbox_xform_clip,
                                    True)


            # view as [N, pre_nms_top_n, 4]
            proposals = proposals.view(N, -1, 4)
            objectness = objectness.view(N, -1)
        else:
            # put in the same format as anchors
            objectness = objectness.permute(0, 2, 3, 1).reshape(N, -1)
            objectness = objectness.sigmoid()
            box_regression = box_regression.view(N, -1, 4, H, W).permute(0, 3, 4, 1, 2)
            box_regression = box_regression.reshape(N, -1, 4)

            num_anchors = A * H * W

            pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)
            objectness, topk_idx = objectness.topk(pre_nms_top_n, dim=1, sorted=True)

            # TODO check if this batch_idx is really needed
            batch_idx = torch.arange(N, device=device)[:, None]
            box_regression = box_regression[batch_idx, topk_idx]

            image_shapes = [box.size[::-1] for box in anchors]
            concat_anchors = torch.cat([a.bbox for a in anchors], dim=0)
            concat_anchors = concat_anchors.reshape(N, -1, 4)[batch_idx, topk_idx]

            proposals = self.box_coder.decode(
                box_regression.view(-1, 4), concat_anchors.view(-1, 4)
            )

            proposals = proposals.view(N, -1, 4)

        # handle non-optimized path without changing loop
        if not use_fast_cuda_path:
            keep = [None for _ in range(num_images)]

        # TODO optimize / make batch friendly
        sampled_bboxes = []
        for proposal, score, im_shape, k in zip(proposals, objectness, image_shapes, keep):
            height, width = im_shape

            if proposal.dim() == 0:
                # TODO check what to do here
                # sampled_proposals.append(proposal.new())
                # sampled_scores.append(score.new())
                print("skipping")
                continue

            if False: # currently slower
                # TODO: Don't do this, generate k directly in bytes
                k = k.byte()
                proposal = proposal[k, :]
                score = score[k]

                # perform NMS - returns index mask of kept boxes
                if self.nms_thresh > 0:
                    keep_mask = C.nms_gpu_upright(proposal, pre_nms_top_n, self.nms_thresh)

                # keep map should still be ordered by score - keep only the post_nms_top_n entries
                if self.post_nms_top_n > 0:
                    keep_mask = keep_mask[:self.post_nms_top_n]

                # keep only selected boxes & scores
                keep_mask = keep_mask.long()
                p = proposal[keep_mask, :]
                score = score[keep_mask]
            else:
                if use_fast_cuda_path:
                    k = k.byte()
                    p = proposal.masked_select(k[:, None]).view(-1, 4)
                    score = score.masked_select(k)
                if self.nms_thresh > 0:
                    keep = box_nms(p, score, self.nms_thresh)
                    if self.post_nms_top_n > 0:
                        keep = keep[: self.post_nms_top_n]
                    p = p.index_select(0, keep)
                    score = score.index_select(0, keep)

            # Common code path
            sampled_bbox = BoxList(p, (width, height), mode="xyxy")
            sampled_bbox.add_field("objectness", score)
            sampled_bboxes.append(sampled_bbox)
            # TODO maybe also copy the other fields that were originally present?

        return sampled_bboxes

    def forward(self, anchors, objectness, box_regression, targets = None):
        """
        Arguments:
            anchors: list[list[BoxList]]
            objectness: list[tensor]
            box_regression: list[tensor]
        """
        assert len(anchors) == 1, "only single feature map supported"
        sampled_boxes = []
        for a, o, b in zip(anchors, objectness, box_regression):
            sampled_boxes.append(self.forward_for_single_feature_map(a, o, b))

        # append ground-truth bboxes to proposals
        if self.training and targets is not None:
            sampled_boxes, _, _ = self.add_gt_proposals(
                    sampled_boxes,
                    indices = None,
                    extra_fields = None,
                    targets = targets)

        return sampled_boxes


class FPNRPNBoxSelector(RPNBoxSelector):
    def __init__(self, roi_to_fpn_level_mapper, fpn_post_nms_top_n, **kwargs):
        """
        Arguments:
            roi_to_fpn_level_mapper (ROI2FPNLevelsMapper)
            fpn_post_nms_top_n (int)
            + same arguments as RPNBoxSelector
        """
        super(FPNRPNBoxSelector, self).__init__(**kwargs)
        self.roi_to_fpn_level_mapper = roi_to_fpn_level_mapper
        self.fpn_post_nms_top_n = fpn_post_nms_top_n

    def __call__(self, anchors, objectness, box_regression, targets=None):
        """
        Arguments:
            anchors: list[list[BoxList]]
            objectness: list[tensor]
            box_regression: list[tensor]
        """
        sampled_boxes = []
        for a, o, b in zip(anchors, objectness, box_regression):
            sampled_boxes.append(self.forward_for_single_feature_map(a, o, b))

        # shortcut for single feature maps
        if len(sampled_boxes) == 1:
            return sampled_boxes

        # TODO almost all this part can be
        # factored out in a RPN-agnostic class
        # merge all lists
        num_features = len(sampled_boxes)
        num_images = len(sampled_boxes[0])

        merged_lists = [
            box for per_feature_boxes in sampled_boxes for box in per_feature_boxes
        ]
        image_sizes = [b.size for b in sampled_boxes[0]]

        device = merged_lists[0].bbox.device
        indices = [
            torch.full((box.bbox.shape[0],), img_idx, device=device)
            for per_feature_boxes in sampled_boxes
            for img_idx, box in enumerate(per_feature_boxes)
        ]

        # TODO make these concatenations a helper function?
        # once we make train and test consistent, this can be greatly
        # simplified, as it will go per image, and not per batch
        concat_boxes = torch.cat([b.bbox for b in merged_lists], dim=0)
        indices = torch.cat(indices, dim=0)
        extra_fields = {}
        field_names = merged_lists[0].fields()
        for field in field_names:
            extra_fields[field] = torch.cat(
                [b.get_field(field) for b in merged_lists], dim=0
            )

        post_nms_top_n = min(self.fpn_post_nms_top_n, concat_boxes.shape[0])
        # different behavior during training and during testing:
        # during training, post_nms_top_n is over *all* the proposals combined, while
        # during testing, it is over the proposals for each image
        # TODO resolve this difference and make it consistent. It should be per image,
        # and not per batch
        if self.training:
            _, inds_sorted = torch.topk(
                extra_fields["objectness"], post_nms_top_n, dim=0, sorted=True
            )
        else:
            inds_sorted = []
            for i in range(num_images):
                objectness = extra_fields["objectness"].clone()
                objectness[indices != i] = -1
                _, inds_sorted_img = torch.topk(
                    objectness, post_nms_top_n, dim=0, sorted=True
                )
                inds_sorted.append(inds_sorted_img)
            inds_sorted = cat(inds_sorted, dim=0)

        concat_boxes = concat_boxes[inds_sorted]
        indices = indices[inds_sorted]
        for field, data in extra_fields.items():
            extra_fields[field] = data[inds_sorted]

        # append ground-truth bboxes to proposals
        if self.training and targets is not None:
            concat_boxes, indices, extra_fields = self.add_gt_proposals(
                    concat_boxes,
                    indices,
                    extra_fields,
                    targets)

        levels = self.roi_to_fpn_level_mapper(concat_boxes)

        # maps back to the original order
        boxes = []
        lvl_min = self.roi_to_fpn_level_mapper.k_min
        lvl_max = self.roi_to_fpn_level_mapper.k_max
        for feat_lvl in range(lvl_min, lvl_max + 1):
            per_feat_boxes = []
            for img_idx in range(num_images):
                lvl_idx_per_img = nonzero((indices == img_idx) & (levels == feat_lvl))[
                    0
                ]
                selected_boxes = concat_boxes.index_select(0,lvl_idx_per_img)
                bbox = BoxList(selected_boxes, image_sizes[img_idx], mode="xyxy")
                for field, data in extra_fields.items():
                    bbox.add_field(field, data.index_select(0,lvl_idx_per_img))
                per_feat_boxes.append(bbox)
            boxes.append(per_feat_boxes)

        return boxes


# TODO move this to bounding box class?
def _clip_boxes_to_image(boxes, height, width):
    fact = 1  # TODO REMOVE
    num_boxes = boxes.shape[0]
    b1 = boxes[:, 0::2].clamp(min=0, max=width - fact)
    b2 = boxes[:, 1::2].clamp(min=0, max=height - fact)
    boxes = torch.stack((b1, b2), 2).view(num_boxes, -1)
    return boxes


def _filter_boxes(boxes, min_size, im_shape):
    """Only keep boxes with both sides >= min_size and center within the image.
    """
    # Scale min_size to match image scale
    fact = 1  # TODO remove
    ws = boxes[:, 2] - boxes[:, 0] + fact
    hs = boxes[:, 3] - boxes[:, 1] + fact
    x_ctr = boxes[:, 0] + ws / 2.
    y_ctr = boxes[:, 1] + hs / 2.
    keep = nonzero(
        (ws >= min_size)
        & (hs >= min_size)
        & (x_ctr < im_shape[1])
        & (y_ctr < im_shape[0])
    )[0]
    return keep


class ROI2FPNLevelsMapper(object):
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.
    """

    def __init__(self, k_min, k_max, canonical_scale=224, canonical_level=4, eps=1e-6):
        """
        Arguments:
            k_min (int)
            k_max (int)
            canonical_scale (int)
            canonical_level (int)
            eps (float)
        """
        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def __call__(self, rois):
        """
        Arguments:
            rois: tensor
        """
        # Compute level ids
        s = torch.sqrt(boxes_area(rois))

        # Eqn.(1) in FPN paper
        target_lvls = torch.floor(self.lvl0 + torch.log2(s / self.s0 + 1e-6))
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        return target_lvls
