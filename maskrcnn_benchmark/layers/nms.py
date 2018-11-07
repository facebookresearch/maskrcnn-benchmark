# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

# from ._utils import _C
from maskrcnn_benchmark import _C

try:
    from apex import amp
    use_apex_amp = True
except ImportError:
    use_apex_amp = False

# Monkey patch in need for fp32
def nms_impl(dets, scores, threshold):
    return _C.nms(dets, scores, threshold)

if use_apex_amp:
    nms = amp.float_function(nms_impl)
else:
    nms = _C.nms

# nms.__doc__ = """
# This function performs Non-maximum suppresion"""
