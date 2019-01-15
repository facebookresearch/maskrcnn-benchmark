"""
Group Normalization Layer from PANet
url: https://github.com/ShuLiu1993/PANet
"""

import torch
import torch.nn as nn
from maskrcnn_benchmark.config import cfg


class GroupNorm(nn.Module):

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_channels))
            self.bias = nn.Parameter(torch.Tensor(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

    def forward(self, x):
        return group_norm(
            x, self.num_groups, self.weight, self.bias, self.eps
        )

    def extra_repr(self):
        return '{num_groups}, {num_channels}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)


def group_norm(x, num_groups, weight=None, bias=None, eps=1e-5):
    input_shape = x.shape
    ndim = len(input_shape)
    N, C = input_shape[:2]
    G = num_groups

    assert C % G == 0, "input channel dimension must divisible by number of groups"
    
    x = x.view(N, G, -1)
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, keepdim=True)
    x = (x - mean) / (var + eps).sqrt()
    x = x.view(input_shape)

    view_shape = (1, -1) + (1,) * (ndim - 2)
    if weight is not None:
        return x * weight.view(view_shape) + bias.view(view_shape)

    return x


def get_group_gn(dim, dim_per_gp, num_groups):
    """get number of groups used by GroupNorm, based on number of channels."""
    assert dim_per_gp == -1 or num_groups == -1, \
        "GroupNorm: can only specify G or C/G."

    if dim_per_gp > 0:
        assert dim % dim_per_gp == 0, \
            "dim: {}, dim_per_gp: {}".format(dim, dim_per_gp)
        group_gn = dim // dim_per_gp
    else:
        assert dim % num_groups == 0, \
            "dim: {}, num_groups: {}".format(dim, num_groups)
        group_gn = num_groups

    return group_gn


def gn_layer_from_cfg(out_channels, affine=True, divisor=1):
    out_channels = out_channels // divisor
    dim_per_gp = cfg.GROUP_NORM.DIM_PER_GP // divisor
    num_groups = cfg.GROUP_NORM.NUM_GROUPS // divisor
    eps = cfg.GROUP_NORM.EPSILON # default: 1e-5
    return GroupNorm(
        get_group_gn(out_channels, dim_per_gp, num_groups), 
        out_channels, 
        eps, 
        affine
    )


