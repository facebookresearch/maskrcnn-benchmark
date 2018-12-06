import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import Conv2d, ConvTranspose2d


class PoseRCNNC4Predictor(nn.Module):
    def __init__(self, cfg):
        super(PoseRCNNC4Predictor, self).__init__()
        stage_index = 4
        stage2_relative_factor = 2 ** (stage_index - 1)
        res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        num_inputs = res2_out_channels * stage2_relative_factor

        num_units = cfg.MODEL.ROI_POSE_HEAD.CONV_LAYERS[-1]

        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES

        # self.conv5_mask = Conv2d(num_inputs, num_units, 1, 1, 0)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
        self.poses_fc1 = nn.Linear(num_inputs, 512)
        self.poses_fc2 = nn.Linear(512, num_classes * 4)

        self._init_params()

    def _init_params(self):
        # nn.init.constant_(self.conv5_mask.bias, 0)
        # nn.init.kaiming_normal_(self.conv5_mask.weight, mode="fan_out", nonlinearity="relu")

        nn.init.normal_(self.poses_fc1.weight, mean=0, std=0.001)
        nn.init.constant_(self.poses_fc1.bias, 0)
        nn.init.normal_(self.poses_fc2.weight, mean=0, std=0.001)
        nn.init.constant_(self.poses_fc2.bias, 0)

    def forward(self, x):
        # x = F.relu(self.conv5_mask(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # pose_pred = self.pose_pred(x)
        fc1 = self.poses_fc1(x)
        fc1 = F.normalize(fc1, p=2, dim=1)
        fc1 = F.dropout(F.relu(fc1, inplace=True), 0.5, training=self.training)
        fc2 = self.poses_fc2(fc1)
        fc2 = F.normalize(fc2, p=2, dim=1)

        return torch.tanh(fc2)


def make_roi_pose_predictor(cfg):
    return PoseRCNNC4Predictor(cfg)
