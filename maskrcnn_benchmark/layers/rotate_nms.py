# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from ._utils import _C
import torch
from torch import nn
from torch.autograd import Function
# from torch.autograd.function import once_differentiable

from maskrcnn_benchmark import _Custom as _C

from apex import amp

rotate_nms = amp.float_function(_C.rotate_nms)
rotate_soft_nms = amp.float_function(_C.rotate_soft_nms)
rotate_iou_matrix = amp.float_function(_C.rotate_iou_matrix)


class _RotateNMSFunction(Function):
    @staticmethod
    def forward(ctx, r_boxes, scores, nms_threshold, post_nms_top_n=-1):
        # r_boxes: (N,5)
        assert len(r_boxes.shape) == 2 and r_boxes.size(1) == 5

        keep_inds = rotate_nms(r_boxes, scores, nms_threshold)

        if post_nms_top_n > 0:
            keep_inds = keep_inds[:post_nms_top_n]
        return keep_inds


def rotate_soft_nms_func(r_boxes, scores, nms_thresh=0.3, sigma=0.5, score_thresh=0.001, method=1, post_nms_top_n=-1):
        # r_boxes: (N,5)
        assert len(r_boxes.shape) == 2 and r_boxes.size(1) == 5 and r_boxes.size(0) == scores.size(0)

        boxes2 = r_boxes.clone()
        scores2 = scores.clone()
        indices, keep = rotate_soft_nms(boxes2, scores2, nms_thresh,
            sigma, score_thresh, method)

        if post_nms_top_n > 0:
            keep = keep[:post_nms_top_n]
        return indices, keep, scores2


class RotateNMS(nn.Module):
    """
    Performs rotated NMS

    INPUT:
    r_boxes: (N, 5)  [xc,yc,w,h,theta]  # NOTE there's no score field here
    scores: (N) 
    """
    def __init__(self, nms_threshold=0.7, post_nms_top_n=-1):
        """
        param: post_nms_top_n < 0 means no max cap
        """
        super(RotateNMS, self).__init__()

        self.nms_threshold = float(nms_threshold)
        self.post_nms_top_n = int(post_nms_top_n)

    def forward(self, r_boxes, scores):
        """
        r_boxes: (N, 5)  [xc,yc,w,h,theta]
        """
        return _RotateNMSFunction.apply(r_boxes, scores, self.nms_threshold, self.post_nms_top_n)

    def __repr__(self):
        tmpstr = "%s (nms_thresh=%.2f, post_nms_top_n=%d)"%(self.__class__.__name__,
                    self.nms_threshold, self.post_nms_top_n)
        return tmpstr


class RotateSoftNMS(nn.Module):
    """
    Performs rotated soft NMS

    INPUT:
    r_boxes: (N, 5)  [xc,yc,w,h,theta]  # NOTE there's no score field here
    scores: (N) 
    """
    def __init__(self, nms_thresh=0.3, sigma=0.5, score_thresh=0.001, method=1, post_nms_top_n=-1):
        super(RotateSoftNMS, self).__init__()

        self.nms_threshold = float(nms_thresh)
        self.sigma = float(sigma)
        self.score_thresh = float(score_thresh)
        self.method = int(method)
        self.post_nms_top_n = int(post_nms_top_n)

    def forward(self, boxes, scores):
        """
        INPUT: 
        boxes: (N, 5)  [xc,yc,w,h,theta]
        scores: N

        Returns:
        sorted_indices: N   # sorted index, by score
        keep: range [0, N]  # indices to keep, AFTER sorted_indices applied
        scores_new: N  # new scores after soft nms, AFTER sorted_indices applied
        """
        sorted_indices, keep, scores_new = rotate_soft_nms_func(
            boxes, scores, self.nms_threshold,
            self.sigma, self.score_thresh, self.method, self.post_nms_top_n
        )
        return sorted_indices, keep, scores_new

    def __repr__(self):
        tmpstr = "%s (nms_thresh=%.2f, sigma=%.3f, score_thresh=%.3f, method=%d, post_nms_top_n=%d)" % \
        (
            self.__class__.__name__,
            self.nms_threshold, self.sigma, self.score_thresh, self.method, self.post_nms_top_n
        )
        return tmpstr


def rotate_iou(boxes1, boxes2):
    # N = boxes1.size(0)
    assert len(boxes1.shape) == 2 and len(boxes2.shape) == 2 \
           and boxes1.size(1) == 5 and boxes2.size(1) == 5

    iou_matrix = rotate_iou_matrix(boxes1, boxes2)
    return iou_matrix
