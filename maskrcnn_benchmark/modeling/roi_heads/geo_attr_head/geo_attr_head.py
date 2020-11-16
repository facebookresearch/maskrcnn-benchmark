
import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList

from .roi_geo_attr_feature_extractors import make_roi_geo_attr_feature_extractor
from .roi_geo_attr_predictors import make_roi_geo_attr_predictor
# from .inference import make_roi_geo_attr_post_processor
from .loss import make_roi_geo_attr_loss_evaluator

def keep_only_positive_boxes(boxes):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field("labels")
    positive_boxes = []
    positive_inds = []
    num_boxes = 0
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
        positive_inds.append(inds_mask)
    return positive_boxes, positive_inds

class ROIGEOATTRHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ROIGEOATTRHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_geo_attr_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_geo_attr_predictor(
            cfg, self.feature_extractor.out_channels)
        # self.post_processor = make_roi_geo_attr_post_processor(cfg)
        self.loss_evaluator = make_roi_geo_attr_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        if self.training:
            all_proposals = proposals
            proposals, positive_inds = keep_only_positive_boxes(proposals)

        if self.training and self.cfg.MODEL.ROI_GEO_ATTR_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            x = features
            x = x[torch.cat(positive_inds, dim=0)]
        else:
            x = self.feature_extractor(features, proposals)

        geo_attr_logits = self.predictor(x)

        # if not self.training:
            # result = self.post_processor(geo_attr_logits, proposals)
            # return x, result, {}

        loss_geo_attr = self.loss_evaluator(proposals, geo_attr_logits, targets)
        return x, all_proposals, loss_geo_attr

def build_roi_geo_attr_head(cfg, in_channels):
    return ROIGEOATTRHead(cfg, in_channels)
