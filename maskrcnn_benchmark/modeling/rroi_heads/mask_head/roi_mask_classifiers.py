from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import ConvTranspose2d
# from maskrcnn_benchmark.modeling import registry


# @registry.ROI_MASK_PREDICTOR.register("MaskRCNNC4Classifier")
class MaskRCNNC4Classifier(nn.Module):
    def __init__(self, cfg, in_channels):
        super(MaskRCNNC4Classifier, self).__init__()
        num_inputs = in_channels

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_score = nn.Linear(num_inputs, 1)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        cls_logit = self.cls_score(x)
        return cls_logit


# def make_roi_mask_predictor(cfg, in_channels):
#     func = registry.ROI_MASK_PREDICTOR[cfg.MODEL.ROI_MASK_HEAD.PREDICTOR]
#     return func(cfg, in_channels)
