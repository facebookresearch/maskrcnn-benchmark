# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math

import torch
import numpy as np

EPSILON = 1e-8


def stack(x, dim=0, lib=np):
    if lib == np:
        return lib.stack(x, axis=dim)
    elif lib == torch:
        return lib.stack(x, dim=dim)
    else:
        raise NotImplementedError


def clamp(x, min=None, max=None, lib=np):
    if lib == np:
        return lib.clip(x, a_min=min, a_max=max)
    elif lib == torch:
        return lib.clamp(x, min=min, max=max)
    else:
        raise NotImplementedError


class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights=None, bbox_xform_clip=np.log(1000. / 16), lib=torch):
        """
        Arguments:
            weights (5-element tuple)  # None or xc,yc,w,h,theta
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip
        if not (lib == np or lib == torch):
            raise NotImplementedError

        self.lib = lib

    def encode(self, unencode_boxes, reference_boxes):  # np.ones(5, dtype=np.float32)):
        '''
        :param unencode_boxes: [batch_size*H*W*num_anchors_per_location, 5]
        :param reference_boxes: [H*W*num_anchors_per_location, 5]
        :return: encode_boxes [-1, 5]  # xc,yc,w,h,theta
        '''
        weights = self.weights
        lib = self.lib

        x_center, y_center, w, h, theta = \
            unencode_boxes[:, 0], unencode_boxes[:, 1], unencode_boxes[:, 2], unencode_boxes[:, 3], unencode_boxes[:, 4]
        reference_x_center, reference_y_center, reference_w, reference_h, reference_theta = \
            reference_boxes[:, 0], reference_boxes[:, 1], reference_boxes[:, 2], reference_boxes[:, 3], reference_boxes[
                                                                                                        :, 4]

        reference_w += EPSILON
        reference_h += EPSILON
        w += EPSILON
        h += EPSILON  # to avoid NaN in division and log below
        t_xcenter = (x_center - reference_x_center) / reference_w
        t_ycenter = (y_center - reference_y_center) / reference_h
        t_w = lib.log(w / reference_w)
        t_h = lib.log(h / reference_h)

        """
        TO PREVENT angle AMBIGUITY
        for targets where the height and width are roughly similar, there may be ambiguity in angle regression
        e.g. if height and width are equal, angle regression could be -90 or 0 degrees
        we don't want to penalize this
        #
        """
        # THRESH = 0.15
        # w_to_h_ratio = w / h
        # w_to_h_ratio_diff = self.lib.abs(1.0 - w_to_h_ratio)
        # adj_theta = theta.clone() if self.lib == torch else theta.copy()
        # square_ids = w_to_h_ratio_diff < THRESH
        # adj_squares_theta = adj_theta[square_ids]
        # adj_squares_theta[adj_squares_theta > 90] -= 90
        # adj_squares_theta[adj_squares_theta > 45] -= 90
        # adj_theta[square_ids] = adj_squares_theta

        t_theta = theta - reference_theta
        # t_theta[t_theta > 90] -= 90
        # t_theta[t_theta > 45] -= 90

        t_theta = t_theta * np.pi / 180  # convert to radians

        if weights is not None:
            wx, wy, ww, wh, wa = weights
            t_xcenter *= wx
            t_ycenter *= wy
            t_w *= ww
            t_h *= wh
            t_theta *= wa

        encode_boxes = stack([t_xcenter, t_ycenter, t_w, t_h, t_theta], dim=1, lib=lib)

        return encode_boxes

    def decode(self, encode_boxes, reference_boxes):
        '''
        :param encode_boxes:[N, 5]  # xc,yc,w,h,theta
        :param reference_boxes: [N, 5] # xc,yc,w,h,theta
        :param scale_factors: use for scale
        in the rpn stage, reference_boxes are anchors
        in the fast_rcnn stage, reference boxes are proposals(decode) produced by rpn stage
        :return:decode boxes [N, 5]
        '''
        weights = self.weights
        lib = self.lib

        t_xcenter = encode_boxes[:, 0]
        t_ycenter = encode_boxes[:, 1]
        t_w = encode_boxes[:, 2]
        t_h = encode_boxes[:, 3]
        t_theta = encode_boxes[:, 4]

        if weights is not None:
            wx, wy, ww, wh, wa = weights
            t_xcenter /= wx
            t_ycenter /= wy
            t_w /= ww
            t_h /= wh
            t_theta /= wa

        dw = clamp(t_w, max=self.bbox_xform_clip, lib=lib)
        dh = clamp(t_h, max=self.bbox_xform_clip, lib=lib)

        reference_x_center = reference_boxes[:, 0]
        reference_y_center = reference_boxes[:, 1]
        reference_w = reference_boxes[:, 2]
        reference_h = reference_boxes[:, 3]
        reference_theta = reference_boxes[:, 4]

        predict_x_center = t_xcenter * reference_w + reference_x_center
        predict_y_center = t_ycenter * reference_h + reference_y_center
        predict_w = lib.exp(dw) * reference_w
        predict_h = lib.exp(dh) * reference_h
        predict_theta = t_theta * 180 / np.pi + reference_theta  # radians to degrees

        decode_boxes = stack([predict_x_center, predict_y_center, predict_w, predict_h, predict_theta], dim=1, lib=lib)

        return decode_boxes