# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from ._utils import _C
import torch
from torch import nn
from torch.autograd import Function
# from torch.autograd.function import once_differentiable

from maskrcnn_benchmark import _Custom as _C

from apex import amp

rotate_nms = amp.float_function(_C.rotate_nms)
rotate_iou_matrix = amp.float_function(_C.rotate_iou_matrix)

class _RotateNMSFunction(Function):
    @staticmethod
    def forward(ctx, r_boxes, nms_threshold, post_nms_top_n):
        # r_boxes: (N,5)
        assert len(r_boxes.shape) == 2 and r_boxes.size(1) == 5

        keep_inds = rotate_nms(r_boxes, nms_threshold, post_nms_top_n)

        return keep_inds

class RotateNMS(nn.Module):
    """
    Performs rotated NMS
    DOES NOT PERFORM SORTING ON A 'score/objectness' field.
    ASSUMES THE INPUT BOXES ARE ALREADY SORTED

    INPUT:
    r_boxes: (N, 5)  [xc,yc,w,h,theta]  # NOTE there's no score field here
    """
    def __init__(self, nms_threshold=0.7, post_nms_top_n=-1):
        """
        param: post_nms_top_n < 0 means no max cap
        """
        super(RotateNMS, self).__init__()

        self.nms_threshold = float(nms_threshold)
        self.post_nms_top_n = int(post_nms_top_n)

    def forward(self, r_boxes):
        """
        r_boxes: (N, 5)  [xc,yc,w,h,theta]
        """
        return _RotateNMSFunction.apply(r_boxes, self.nms_threshold, self.post_nms_top_n)

    def __repr__(self):
        tmpstr = "%s (nms_thresh=%.2f, post_nms_top_n=%d)"%(self.__class__.__name__,
                    self.nms_threshold, self.post_nms_top_n)
        return tmpstr

def rotate_iou(boxes1, boxes2):
    # N = boxes1.size(0)
    assert len(boxes1.shape) == 2 and len(boxes2.shape) == 2 \
           and boxes1.size(1) == 5 and boxes2.size(1) == 5

    iou_matrix = rotate_iou_matrix(boxes1, boxes2)
    return iou_matrix