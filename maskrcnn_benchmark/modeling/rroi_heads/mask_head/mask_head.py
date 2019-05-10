import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList

from maskrcnn_benchmark.modeling.roi_heads.mask_head.roi_mask_feature_extractors import make_roi_mask_feature_extractor
from maskrcnn_benchmark.modeling.roi_heads.mask_head.roi_mask_predictors import make_roi_mask_predictor
from maskrcnn_benchmark.modeling.roi_heads.mask_head.mask_head import ROIMaskHead
from .inference import make_roi_mask_post_processor
from .loss import make_roi_mask_loss_evaluator


class RROIMaskHead(ROIMaskHead):
    def __init__(self, cfg, in_channels):
        super(ROIMaskHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_mask_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_mask_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_mask_post_processor(cfg)
        self.loss_evaluator = make_roi_mask_loss_evaluator(cfg)


def build_roi_mask_head(cfg, in_channels):
    return RROIMaskHead(cfg, in_channels)
