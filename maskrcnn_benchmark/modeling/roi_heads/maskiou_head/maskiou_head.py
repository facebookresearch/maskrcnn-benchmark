# Mask Scoring R-CNN
# Wriiten by zhaojin.huang, 2018-12.

import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList
from .roi_maskiou_feature_extractors import make_roi_maskiou_feature_extractor
from .roi_maskiou_predictors import make_roi_maskiou_predictor
from .inference import make_roi_maskiou_post_processor
from .loss import make_roi_maskiou_loss_evaluator


class ROIMaskIoUHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ROIMaskIoUHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_maskiou_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_maskiou_predictor(cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_maskiou_post_processor(cfg)
        self.loss_evaluator = make_roi_maskiou_loss_evaluator(cfg)

    def forward(self, features, proposals, selected_mask, labels, maskiou_targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            selected_mask (list[Tensor]): targeted mask
            labels (list[Tensor]): class label of mask
            maskiou_targets (list[Tensor], optional): the ground-truth maskiou targets.

        Returns:
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
            results (list[BoxList]): during training, returns None. During testing, the predicted boxlists are returned.
                with the `mask` field set
        """
        if features.shape[0] == 0 and not self.training:
            return {}, proposals

        x = self.feature_extractor(features, selected_mask)
        pred_maskiou = self.predictor(x)

        if not self.training:
            boxes_per_image = [len(box) for box in proposals]
            pred_maskiou = pred_maskiou.split(boxes_per_image, dim=0)
            labels = labels.split(boxes_per_image, dim=0)

            result = self.post_processor(proposals, pred_maskiou, labels)
            return {}, result

        loss_maskiou = self.loss_evaluator(labels, pred_maskiou, maskiou_targets)

        return dict(loss_maskiou=loss_maskiou), proposals


def build_roi_maskiou_head(cfg, in_channels):
    return ROIMaskIoUHead(cfg, in_channels)
