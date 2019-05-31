# Mask Scoring R-CNN
# Wriiten by zhaojin.huang, 2018-12.

import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.modeling.make_layers import make_conv3x3
from maskrcnn_benchmark.modeling.make_layers import make_fc


class MaskIoUFeatureExtractor(nn.Module):
    """
    MaskIou head feature extractor.
    """

    def __init__(self, cfg, in_channels):
        super(MaskIoUFeatureExtractor, self).__init__()
        
        input_channels = in_channels + 1  # cat features and mask single channel
        use_gn = cfg.MODEL.ROI_MASKIOU_HEAD.USE_GN
        representation_size = cfg.MODEL.ROI_MASKIOU_HEAD.MLP_HEAD_DIM

        input_resolution = cfg.MODEL.ROI_MASK_HEAD.RESOLUTION
        resolution = input_resolution // 2  # after max pooling 2x2
        layers = cfg.MODEL.ROI_MASKIOU_HEAD.CONV_LAYERS
        # stride=1 for each layer, and stride=2 for last layer
        strides = [1 for l in layers]
        strides[-1] = 2

        next_feature = input_channels
        self.blocks = []
        for layer_idx, layer_features in enumerate(layers):
            layer_name = "maskiou_fcn{}".format(layer_idx+1)
            stride = strides[layer_idx]
            module = make_conv3x3(next_feature, layer_features, stride=stride, dilation=1, use_gn=use_gn)
            self.add_module(layer_name, module)
            self.blocks.append(layer_name)

            next_feature = layer_features
            if stride == 2:
                resolution = resolution // 2

        self.maskiou_fc1 = make_fc(next_feature*resolution**2, representation_size, use_gn=False)
        self.maskiou_fc2 = make_fc(representation_size, representation_size, use_gn=False)
        self.out_channels = representation_size


    def forward(self, x, mask):
        mask_pool = F.max_pool2d(mask, kernel_size=2, stride=2)
        x = torch.cat((x, mask_pool), 1)

        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.maskiou_fc1(x))
        x = F.relu(self.maskiou_fc2(x))
 
        return x


def make_roi_maskiou_feature_extractor(cfg, in_channels):
    func = MaskIoUFeatureExtractor
    return func(cfg, in_channels)
