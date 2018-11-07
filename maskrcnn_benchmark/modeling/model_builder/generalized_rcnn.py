# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

"""
Proposal implementation for the model builder
Everything that constructs something is a function that is
or can be accessed by a string
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from maskrcnn_benchmark.modeling.box_coder import BoxCoder

from maskrcnn_benchmark.modeling.post_processors.fast_rcnn import (
    PostProcessor,
    FPNPostProcessor,
)
from maskrcnn_benchmark.modeling.post_processors.mask_rcnn import MaskPostProcessor

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .roi_mask_feature_extractors import make_roi_mask_feature_extractor
from .roi_mask_predictors import make_roi_mask_predictor
from .backbones import build_backbone

from .rpn import build_rpn
from . import loss_evaluators


class GeneralizedRCNN(nn.Module):
    """
    Main class for generalized r-cnn. Supports boxes, masks and keypoints
    This is very similar to what we had before, the difference being that now
    we construct the modules in __init__, instead of passing them as arguments
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.cfg = cfg.clone()

        self.backbone = build_backbone(cfg)

        self.rpn = build_rpn(cfg)

        # individually create the heads, that will be combined together
        # afterwards
        roi_heads = []
        if not cfg.MODEL.RPN_ONLY:
            roi_heads.append(build_roi_box_head(cfg))
        if cfg.MODEL.MASK_ON:
            roi_heads.append(build_roi_mask_head(cfg))

        # combine individual heads in a single module
        if roi_heads:
            self.roi_heads = combine_roi_heads(cfg, roi_heads)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.cfg.MODEL.RPN_ONLY:
            x = features
            result = proposals
            detector_losses = {}
        else:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result


_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)


################################################################################
# Those generic classes shows that it's possible to plug both Mask R-CNN C4
# and Mask R-CNN FPN with generic containers
################################################################################
class CascadedHeads(torch.nn.Module):
    """
    For Mask R-CNN FPN
    """

    def __init__(self, heads):
        super(CascadedHeads, self).__init__()
        self.heads = torch.nn.ModuleList(heads)

    def forward(self, features, proposals, targets=None):
        losses = {}
        for head in self.heads:
            x, proposals, loss = head(features, proposals, targets)
            losses.update(loss)

        return x, proposals, losses


class SharedROIHeads(torch.nn.Module):
    """
    For Mask R-CNN C4
    """

    def __init__(self, heads):
        super(SharedROIHeads, self).__init__()
        self.heads = torch.nn.ModuleList(heads)
        # manually share feature extractor
        shared_feature_extractor = self.heads[0].feature_extractor
        for head in self.heads[1:]:
            head.feature_extractor = shared_feature_extractor

    def forward(self, features, proposals, targets=None):
        losses = {}
        for head in self.heads:
            x, proposals, loss = head(features, proposals, targets)
            # during training, share the features
            if self.training:
                features = x
            losses.update(loss)

        return x, proposals, losses


def combine_roi_heads(cfg, roi_heads):
    """
    This function takes a list of modules for the rois computation and
    combines them into a single head module.
    It also support feature sharing, for example during
    Mask R-CNN C4.
    """
    constructor = CascadedHeads
    if cfg.MODEL.SHARE_FEATURES_DURING_TRAINING:
        constructor = SharedROIHeads
    # can also use getfunc to query a function by name
    return constructor(roi_heads)


################################################################################
# RoiBoxHead
################################################################################


def make_roi_box_post_processor(cfg):
    use_fpn = cfg.MODEL.ROI_HEADS.USE_FPN

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
    nms_thresh = cfg.MODEL.ROI_HEADS.NMS
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG

    postprocessor_maker = PostProcessor if not use_fpn else FPNPostProcessor
    postprocessor = postprocessor_maker(
        score_thresh, nms_thresh, detections_per_img, box_coder
    )
    return postprocessor


class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg)
        self.predictor = make_roi_box_predictor(cfg)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = loss_evaluators.make_roi_box_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        if self.training:
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        x = self.feature_extractor(features, proposals)
        class_logits, box_regression = self.predictor(x)

        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)
            return x, result, {}

        loss_classifier, loss_box_reg = self.loss_evaluator(
            [class_logits], [box_regression]
        )
        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
        )


def build_roi_box_head(cfg):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg)


################################################################################
# RoiMaskHead
################################################################################


def make_roi_mask_post_processor(cfg):
    masker = None
    mask_post_processor = MaskPostProcessor(masker)
    return mask_post_processor


class ROIMaskHead(torch.nn.Module):
    def __init__(self, cfg):
        super(ROIMaskHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_mask_feature_extractor(cfg)
        self.predictor = make_roi_mask_predictor(cfg)
        self.post_processor = make_roi_mask_post_processor(cfg)
        self.loss_evaluator = loss_evaluators.make_roi_mask_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        # the network can't handle the case of no selected proposals,
        # so need to shortcut before
        # TODO handle this properly
        if not self.training:
            if sum(r.bbox.shape[0] for r in proposals) == 0:
                for r in proposals:
                    r.add_field("mask", features[0].new())
                return features, proposals, {}

        if self.training and self.cfg.MODEL.SHARE_FEATURES_DURING_TRAINING:
            x = features
        else:
            if self.cfg.MODEL.SHARE_FEATURES_DURING_TRAINING:
                # proposals have been flattened by this point, so aren't
                # in the format of list[list[BoxList]] anymore. Add one more level to it
                proposals = [proposals]
            x = self.feature_extractor(features, proposals)
        mask_logits = self.predictor(x)

        if not self.training:
            # TODO Fix this horrible hack
            if self.cfg.MODEL.SHARE_FEATURES_DURING_TRAINING:
                proposals = proposals[0]
            result = self.post_processor(mask_logits, proposals)
            return x, result, {}

        loss_mask = self.loss_evaluator(proposals, mask_logits, targets)

        return x, proposals, dict(loss_mask=loss_mask)


def build_roi_mask_head(cfg):
    return ROIMaskHead(cfg)
