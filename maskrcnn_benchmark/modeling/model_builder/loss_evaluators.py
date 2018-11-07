# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
Functions for creating loss evaluators
"""
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.losses.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.losses.fast_rcnn import FastRCNNLossComputation
from maskrcnn_benchmark.modeling.losses.fast_rcnn import FastRCNNTargetPreparator
from maskrcnn_benchmark.modeling.losses.mask_rcnn import MaskRCNNLossComputation
from maskrcnn_benchmark.modeling.losses.mask_rcnn import MaskTargetPreparator
from maskrcnn_benchmark.modeling.losses.matcher import Matcher
from maskrcnn_benchmark.modeling.losses.rpn import RPNLossComputation
from maskrcnn_benchmark.modeling.losses.rpn import RPNTargetPreparator


def make_standard_loss_evaluator(
    loss_type,
    fg_iou_threshold,
    bg_iou_threshold,
    batch_size_per_image=None,
    positive_fraction=None,
    box_coder=None,
    mask_resolution=None,
    mask_subsample_only_positive_boxes=None,
):
    assert loss_type in ("rpn", "fast_rcnn", "mask_rcnn")
    allow_low_quality_matches = loss_type == "rpn"
    matcher = Matcher(
        fg_iou_threshold,
        bg_iou_threshold,
        allow_low_quality_matches=allow_low_quality_matches,
    )

    if loss_type in ("rpn", "fast_rcnn"):
        assert isinstance(batch_size_per_image, int)
        assert isinstance(positive_fraction, (int, float))
        assert isinstance(box_coder, BoxCoder)
        fg_bg_sampler = BalancedPositiveNegativeSampler(
            batch_size_per_image=batch_size_per_image,
            positive_fraction=positive_fraction,
        )

    if loss_type == "rpn":
        for arg in (mask_resolution, mask_subsample_only_positive_boxes):
            assert arg is None
        target_preparator = RPNTargetPreparator(matcher, box_coder)
        loss_evaluator = RPNLossComputation(target_preparator, fg_bg_sampler)
    elif loss_type == "fast_rcnn":
        for arg in (mask_resolution, mask_subsample_only_positive_boxes):
            assert arg is None
        target_preparator = FastRCNNTargetPreparator(matcher, box_coder)
        loss_evaluator = FastRCNNLossComputation(target_preparator, fg_bg_sampler)
    elif loss_type == "mask_rcnn":
        for arg in (batch_size_per_image, positive_fraction):
            assert arg is None
        assert isinstance(mask_resolution, (int, float))
        assert isinstance(mask_subsample_only_positive_boxes, bool)
        target_preparator = MaskTargetPreparator(matcher, mask_resolution)
        loss_evaluator = MaskRCNNLossComputation(
            target_preparator,
            subsample_only_positive_boxes=mask_subsample_only_positive_boxes,
        )

    return loss_evaluator


def make_rpn_loss_evaluator(cfg, rpn_box_coder):
    return make_standard_loss_evaluator(
        "rpn",
        cfg.MODEL.RPN.FG_IOU_THRESHOLD,
        cfg.MODEL.RPN.BG_IOU_THRESHOLD,
        cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE,
        cfg.MODEL.RPN.POSITIVE_FRACTION,
        rpn_box_coder,
    )


def make_roi_box_loss_evaluator(cfg):
    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)
    return make_standard_loss_evaluator(
        "fast_rcnn",
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
        cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION,
        box_coder,
    )


def make_roi_mask_loss_evaluator(cfg):
    return make_standard_loss_evaluator(
        "mask_rcnn",
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        mask_resolution=cfg.MODEL.ROI_MASK_HEAD.RESOLUTION,
        mask_subsample_only_positive_boxes=cfg.MODEL.ROI_HEADS.USE_FPN,
    )
