from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import ConvTranspose2d


def conv_transpose2d_by_factor(in_cn, out_cn, factor, trainable=True):
    """
    Maintain output_size = input_size * factor (multiple of 2)
    """
    # stride = int(1.0/spatial_scale)
    assert factor >= 2 and factor % 2 == 0
    stride = factor
    k = stride * 2
    kernel_size = (k,k)
    padding = stride / 2
    stride = (stride, stride)
    return ConvTranspose2d(in_cn, out_cn, kernel_size, stride, padding, trainable=trainable)


class VertexRCNNC4Predictor(nn.Module):
    def __init__(self, cfg):
        super(VertexRCNNC4Predictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        num_units = cfg.MODEL.ROI_VERTEX_HEAD.CONV_LAYERS[-1]

        if cfg.MODEL.ROI_HEADS.USE_FPN:
            num_inputs = num_units
        else:
            stage_index = 4
            stage2_relative_factor = 2 ** (stage_index - 1)
            res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
            num_inputs = res2_out_channels * stage2_relative_factor

        # spatial_scale = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES[-1]
        # self.conv5_mask = Conv2d(num_inputs, num_units, 1, 1, 0)
        # self.upconv = conv_transpose2d_by_factor(num_units, num_units, int(1.0/spatial_scale))
        # self.vertex_reg = Conv2d(num_units, num_classes * 3, 1, 1, 0)   # x,y,z vertexes

        self.conv5_mask = ConvTranspose2d(num_inputs, num_units, 2, 2, 0)
        self.vertex_reg = Conv2d(num_units, num_classes * 3, 1, 1, 0)   # x,y,z vertexes

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
        x = F.relu(self.conv5_mask(x))
        # x = F.relu(self.upconv(x))
        return self.vertex_reg(x)


# _ROI_VERTEX_PREDICTOR = {"VertexRCNNC4Predictor": VertexRCNNC4Predictor}

def make_roi_vertex_predictor(cfg):
    # func = _ROI_VERTEX_PREDICTOR[cfg.MODEL.ROI_VERTEX_HEAD.PREDICTOR]
    return VertexRCNNC4Predictor(cfg)
