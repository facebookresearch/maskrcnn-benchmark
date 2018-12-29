from torch import nn
from torch.nn import functional as F

import numpy as np

from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import ConvTranspose2d


def conv_transpose2d_by_factor(in_cn, out_cn, factor):
    """
    Maintain output_size = input_size * factor (multiple of 2)
    """
    # stride = int(1.0/spatial_scale)
    assert factor >= 2 and factor % 2 == 0
    stride = factor
    k = stride * 2
    kernel_size = (k,k)
    p = stride // 2
    padding = (p, p)
    stride = (stride, stride)
    return ConvTranspose2d(in_cn, out_cn, kernel_size, stride, padding)


class DepthRCNNC4Predictor(nn.Module):
    def __init__(self, cfg):
        super(DepthRCNNC4Predictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES

        if cfg.MODEL.ROI_HEADS.USE_FPN:
            raise NotImplementedError("Not implemented with FPN!")
        else:
            stage_index = 4
            stage2_relative_factor = 2 ** (stage_index - 1)
            res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
            num_inputs = res2_out_channels * stage2_relative_factor

        upsample_factor = cfg.MODEL.ROI_DEPTH_HEAD.UPSAMPLE_FACTOR
        stages = int(np.ceil(np.log2(upsample_factor)))

        model = nn.Sequential()
        for i in range(stages):
            conv_t = conv_transpose2d_by_factor(num_inputs, 256, factor=2)
            num_inputs = 256
            model.add_module("%d"%(i), conv_t)
            model.add_module("relu_%d" % (i), nn.ReLU())

        self.model = model
        kernel_sz = 5
        self.depth_reg = Conv2d(num_inputs, num_classes, kernel_sz, 1, kernel_sz // 2)

        self._init_params()

    def _init_params(self):
        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
        
    def forward(self, x):
        x = self.model(x)
        return self.depth_reg(x)

def make_roi_depth_predictor(cfg):
    # func = _ROI_VERTEX_PREDICTOR[cfg.MODEL.ROI_VERTEX_HEAD.PREDICTOR]
    return DepthRCNNC4Predictor(cfg)
