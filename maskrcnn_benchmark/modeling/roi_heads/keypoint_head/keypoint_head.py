import torch

from .roi_keypoint_feature_extractors import make_roi_keypoint_feature_extractor
from .roi_keypoint_predictors import make_roi_keypoint_predictor
from .inference import make_roi_keypoint_post_processor
# from .loss import make_roi_keypoint_loss_evaluator


class ROIKeypointHead(torch.nn.Module):
    def __init__(self, cfg):
        super(ROIKeypointHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_keypoint_feature_extractor(cfg)
        self.predictor = make_roi_keypoint_predictor(cfg)
        self.post_processor = make_roi_keypoint_post_processor(cfg)
        # self.loss_evaluator = make_roi_keypoint_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        # the network can't handle the case of no selected proposals,
        # so need to shortcut before
        # TODO handle this properly
        if not self.training:
            if sum(r.bbox.shape[0] for r in proposals) == 0:
                for r in proposals:
                    r.add_field("keypoints", features[0].new())
                return features, proposals, {}

        if self.training:
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        x = self.feature_extractor(features, proposals)
        kp_logits = self.predictor(x)

        if not self.training:
            result = self.post_processor(kp_logits, proposals)
            return x, result, {}

        loss_kp = self.loss_evaluator(proposals, kp_logits)

        return x, proposals, dict(loss_kp=loss_kp)

def build_roi_keypoint_head(cfg):
    return ROIKeypointHead(cfg)
