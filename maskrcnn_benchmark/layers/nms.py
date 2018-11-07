# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from ._utils import _C
import torch

from maskrcnn_benchmark import _C

torch.ops.load_library(_C.__file__)

if 0:  # the easy way
    nms = torch.ops.maskrcnn_benchmark.nms


def nms(dets, scores, threshold):
    """This function performs Non-maximum suppresion"""
    @torch.jit.script
    def _nms1(dets, scores, threshold: float=threshold):
        return torch.ops.maskrcnn_benchmark.nms(dets, scores, threshold)

    @torch.jit.script
    def _nms(dets, scores):
        return _nms1(dets, scores)

    return _nms(dets, scores)
