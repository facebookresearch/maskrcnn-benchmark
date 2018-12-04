from torch import nn
from torch.nn import functional as F


class PoseRCNNC4Predictor(nn.Module):
    def __init__(self, cfg):
        super(PoseRCNNC4Predictor, self).__init__()
        stage_index = 4
        stage2_relative_factor = 2 ** (stage_index - 1)
        res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        num_inputs = res2_out_channels * stage2_relative_factor

        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
        self.pose_pred = nn.Linear(num_inputs, num_classes * 4)

        self._init_params()

    def _init_params(self):
        nn.init.normal_(self.pose_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.pose_pred.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        pose_pred = self.pose_pred(x)
        return pose_pred


def make_roi_pose_predictor(cfg):
    return PoseRCNNC4Predictor(cfg)
