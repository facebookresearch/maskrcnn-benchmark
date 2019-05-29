import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList

from maskrcnn_benchmark.modeling.roi_heads.mask_head.roi_mask_feature_extractors import make_roi_mask_feature_extractor
from maskrcnn_benchmark.modeling.roi_heads.mask_head.roi_mask_predictors import make_roi_mask_predictor
from maskrcnn_benchmark.modeling.roi_heads.mask_head.mask_head import keep_only_positive_boxes
from .inference import make_roi_mask_post_processor
from .loss import make_roi_mask_loss_evaluator
from .roi_mask_classifiers import MaskRCNNC4Classifier


class RROIMaskHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(RROIMaskHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_mask_feature_extractor(cfg, in_channels)

        in_channels = self.feature_extractor.out_channels
        self.predictor = make_roi_mask_predictor(cfg, in_channels)
        self.classifier = lambda x: None
        if self.cfg.MODEL.ROI_MASK_HEAD.WITH_CLASSIFIER:
            self.classifier = MaskRCNNC4Classifier(cfg, in_channels)
        self.post_processor = make_roi_mask_post_processor(cfg)
        self.loss_evaluator = make_roi_mask_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # during training, only focus on positive boxes
            # all_proposals = proposals
            proposals, positive_inds = keep_only_positive_boxes(proposals)
        if self.training and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            x = features
            x = x[torch.cat(positive_inds, dim=0)]
        else:
            x = self.feature_extractor(features, proposals)
        mask_logits = self.predictor(x)
        cls_logits = self.classifier(x)

        if not self.training:
            result = self.post_processor(mask_logits, proposals, cls_logits)
            return x, mask_logits, result, {}

        loss_mask, loss_cls = self.loss_evaluator(proposals, mask_logits, targets, cls_logits)
        losses = dict(loss_mask=loss_mask)
        if cls_logits is not None:
            losses['loss_mask_cls'] = loss_cls

        return x, mask_logits, proposals, losses

def build_roi_mask_head(cfg, in_channels):
    return RROIMaskHead(cfg, in_channels)
