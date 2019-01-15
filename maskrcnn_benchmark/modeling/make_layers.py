# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Miscellaneous utility functions
"""

import torch
from torch import nn
from torch.nn import functional as F
from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import gn_layer_from_cfg
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.backbone import resnet


def make_conv3x3(
    in_chs, out_chs, dilation=1, stride=1, use_gn=False, kaiming_init=True
):
    conv = Conv2d(
        in_chs, 
        out_chs, 
        kernel_size=3, 
        stride=stride, 
        padding=dilation, 
        dilation=dilation, 
        bias=False if use_gn else True
    )
    if kaiming_init:
        nn.init.kaiming_normal_(
            conv.weight, mode="fan_out", nonlinearity="relu"
        )
    else:
        torch.nn.init.normal_(conv.weight, std=0.01)
    if not use_gn:
        nn.init.constant_(conv.bias, 0)
    if use_gn:
        return nn.Sequential(
            conv, gn_layer_from_cfg(out_chs), nn.ReLU(inplace=True)
        )
    return nn.Sequential(conv, nn.ReLU(inplace=True))


def make_fc(dim_in, hidden_dim, use_gn):
    '''
        Caffe2 implementation uses XavierFill, which in fact
        corresponds to kaiming_uniform_ in PyTorch
    '''
    if use_gn:
        fc = nn.Linear(dim_in, hidden_dim, bias=False)
        nn.init.kaiming_uniform_(fc.weight, a=1)
        return nn.Sequential(fc, gn_layer_from_cfg(hidden_dim))
    fc = nn.Linear(dim_in, hidden_dim)
    nn.init.kaiming_uniform_(fc.weight, a=1)
    nn.init.constant_(fc.bias, 0)
    return fc
