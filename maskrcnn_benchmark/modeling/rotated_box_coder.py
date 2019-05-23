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
        if weights is not None:
            lenw = len(weights)
            assert 4 <= lenw <= 5
            if lenw == 4:
                self.weights = (weights[0],weights[1],weights[2],weights[3],1.0)
        else:
            self.weights = (1.0, 1.0, 1.0, 1.0, 1.0)
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

        t_theta = theta# - reference_theta
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

    def decode(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """
        lib = self.lib

        boxes = boxes.to(rel_codes.dtype)

        TO_REMOVE = 1  # TODO remove
        widths = boxes[:, 2]
        heights = boxes[:, 3]
        ctr_x = boxes[:, 0]
        ctr_y = boxes[:, 1]
        reference_theta = boxes[:, 4]

        wx, wy, ww, wh, wtheta = self.weights
        dx = rel_codes[:, 0::5] / wx
        dy = rel_codes[:, 1::5] / wy
        dw = rel_codes[:, 2::5] / ww
        dh = rel_codes[:, 3::5] / wh
        dtheta = rel_codes[:, 4::5] / wtheta

        # Prevent sending too large values into torch.exp()
        dw = clamp(dw, max=self.bbox_xform_clip, lib=lib)
        dh = clamp(dh, max=self.bbox_xform_clip, lib=lib)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = lib.exp(dw) * widths[:, None]
        pred_h = lib.exp(dh) * heights[:, None]
        pred_theta = dtheta * 180 / np.pi #+ reference_theta[:, None]  # radians to degrees

        pred_boxes = lib.zeros_like(rel_codes)
        pred_boxes[:, 0::5] = pred_ctr_x
        pred_boxes[:, 1::5] = pred_ctr_y
        pred_boxes[:, 2::5] = pred_w
        pred_boxes[:, 3::5] = pred_h
        pred_boxes[:, 4::5] = pred_theta

        return pred_boxes
