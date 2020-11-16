from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import ConvTranspose2d
from maskrcnn_benchmark.modeling import registry

@registry.ROI_GEO_ATTR_PREDICTOR.register("GEOATTRRCNNPredictor")
class GEOATTRRCNNPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(GEOATTRRCNNPredictor, self).__init__()
        num_inputs = in_channels
        self.ori_bin = cfg.MODEL.ROI_GEO_ATTR_HEAD.ORI_BIN

        self.dimension_head = nn.Linear(num_inputs, 3)
        self.ori_conf_head = nn.Linear(num_inputs, self.ori_bin)
        self.ori_consin_head = nn.Linear(num_inputs, self.ori_bin * 2)
        # only predict x and y, no z since pred_dim height /2 is equal to location z
        self.location_head = nn.Linear(num_inputs, 2)

        nn.init.normal_(self.dimension_head.weight, std=0.01)
        nn.init.normal_(self.ori_conf_head.weight, std=0.01)
        nn.init.normal_(self.ori_consin_head.weight, std=0.01)
        nn.init.normal_(self.location_head.weight, std=0.01)
        for l in [self.dimension_head, self.ori_conf_head, self.ori_consin_head, self.location_head]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        pred_dims = self.dimension_head(x)
        pred_ori_conf = self.ori_conf_head(x)
        pred_ori_consin = self.ori_consin_head(x).view(-1, self.ori_bin, 2)
        pred_location = self.location_head(x)

        # if self.training:

        return [pred_dims, pred_ori_conf, pred_ori_consin, pred_location]

def make_roi_geo_attr_predictor(cfg, in_channels):
    func = registry.ROI_GEO_ATTR_PREDICTOR[cfg.MODEL.ROI_GEO_ATTR_HEAD.PREDICTOR]
    return func(cfg, in_channels)
