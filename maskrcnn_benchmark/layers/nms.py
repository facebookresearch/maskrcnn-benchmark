# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from ._utils import _C
from maskrcnn_benchmark import _C

from apex import amp

# Only valid with fp32 inputs - give AMP the hint
nms = amp.float_function(_C.nms)

# nms.__doc__ = """
# This function performs Non-maximum suppresion"""

def soft_nms(boxes, scores, nms_thresh=0.3, sigma=0.5, score_thresh=0.001, method=1):
    # method: 1) linear, 2) gaussian, else) original NMS
    boxes2 = boxes.clone()
    scores2 = scores.clone()
    indices, keep = _C.soft_nms(boxes2, scores2, nms_thresh, sigma, score_thresh, method)
    return indices, keep, scores2

