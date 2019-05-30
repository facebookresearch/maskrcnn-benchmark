# Mask Scoring R-CNN
# Wriiten by zhaojin.huang, 2018-12.

import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import Conv2d


class MaskIoUFeatureExtractor(nn.Module):
    """
    MaskIou head feature extractor.
    """

    def __init__(self, cfg):
        super(MaskIoUFeatureExtractor, self).__init__()
        
        input_channels = 257 

        self.maskiou_fcn1 = Conv2d(input_channels, 256, 3, 1, 1) 
        self.maskiou_fcn2 = Conv2d(256, 256, 3, 1, 1) 
        self.maskiou_fcn3 = Conv2d(256, 256, 3, 1, 1) 
        self.maskiou_fcn4 = Conv2d(256, 256, 3, 2, 1) 
        self.maskiou_fc1 = nn.Linear(256*7*7, 1024)
        self.maskiou_fc2 = nn.Linear(1024, 1024)

        for l in [self.maskiou_fcn1, self.maskiou_fcn2, self.maskiou_fcn3, self.maskiou_fcn4]:
            nn.init.kaiming_normal_(l.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(l.bias, 0)

        for l in [self.maskiou_fc1, self.maskiou_fc2]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)


    def forward(self, x, mask):
        mask_pool = F.max_pool2d(mask, kernel_size=2, stride=2)
        x = torch.cat((x, mask_pool), 1)
        x = F.relu(self.maskiou_fcn1(x))
        x = F.relu(self.maskiou_fcn2(x))
        x = F.relu(self.maskiou_fcn3(x))
        x = F.relu(self.maskiou_fcn4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.maskiou_fc1(x))
        x = F.relu(self.maskiou_fc2(x))
 
        return x


def make_roi_maskiou_feature_extractor(cfg):
    func = MaskIoUFeatureExtractor
    return func(cfg)
