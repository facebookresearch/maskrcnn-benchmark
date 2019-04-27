# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from ._utils import _C
from maskrcnn_benchmark import _C
import torch

# we need this for the custom ops to exist
import maskrcnn_benchmark._custom_ops   # noqa: F401

from maskrcnn_benchmark.utils import amp

_nms = torch.ops.maskrcnn_benchmark.nms

# Only valid with fp32 inputs - give AMP the hint
nms = amp.float_function(_nms)

# nms.__doc__ = """
# This function performs Non-maximum suppresion"""
