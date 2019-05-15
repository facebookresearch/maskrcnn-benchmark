# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2

import torch
from torch import nn
from torch.autograd import Function
# from torch.autograd.function import once_differentiable

from maskrcnn_benchmark import _Custom as _C

def nms_rotate_cpu(boxes, iou_threshold, max_output_size):
    EPSILON = 1e-8

    keep = []

    num = boxes.shape[0]

    suppressed = np.zeros((num), dtype=np.int)

    for i in range(num):
        if len(keep) >= max_output_size:
            break

        if suppressed[i] == 1:
            continue
        keep.append(i)
        r1 = ((boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3]), boxes[i, 4])
        area_r1 = boxes[i, 2] * boxes[i, 3]
        for j in range(i + 1, num):
            # if suppressed[i] == 1:
            #     continue
            r2 = ((boxes[j, 0], boxes[j, 1]), (boxes[j, 2], boxes[j, 3]), boxes[j, 4])
            area_r2 = boxes[j, 2] * boxes[j, 3]
            inter = 0.0

            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)

                int_area = cv2.contourArea(order_pts)

                inter = int_area * 1.0 / (area_r1 + area_r2 - int_area + EPSILON)

            if inter >= iou_threshold:
                suppressed[j] = 1

    return np.array(keep, np.int64)

class _RotateNMSFunction(Function):
    @staticmethod
    def forward(ctx, r_boxes, nms_threshold, post_nms_top_n):
        # boxes: (N,5)
        N = r_boxes.size(0)
        assert len(r_boxes.shape) == 2 and r_boxes.size(1) == 5

        # # keep_inds = torch.zeros(N)
        # if r_boxes.is_cuda:
        #     # keep_inds = keep_inds.type(torch.cuda.IntTensor)
        #     keep_inds = _C.rotate_nms(r_boxes, nms_threshold, post_nms_top_n)
        # else:
        #     # with torch.no_grad():
        #     #     keep_inds = nms_rotate_cpu(r_boxes, nms_threshold, post_nms_top_n)
        #     # keep_inds = torch.LongTensor(keep_inds)
        #     raise NotImplementedError("Rotate NMS Forward CPU layer not implemented!")
        keep_inds = _C.rotate_nms(r_boxes, nms_threshold, post_nms_top_n)

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

def iou_rotate_cpu(boxes1, boxes2):
    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]
    ious = []
    for i, box1 in enumerate(boxes1):
        temp_ious = []
        r1 = ((box1[0], box1[1]), (box1[2], box1[3]), box1[4])
        for j, box2 in enumerate(boxes2):
            r2 = ((box2[0], box2[1]), (box2[2], box2[3]), box2[4])

            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)

                int_area = cv2.contourArea(order_pts)

                inter = int_area * 1.0 / (area1[i] + area2[j] - int_area)
                temp_ious.append(inter)
            else:
                temp_ious.append(0.0)
        ious.append(temp_ious)

    return np.array(ious, dtype=np.float32)

def rotate_iou(boxes1, boxes2):
    # N = boxes1.size(0)
    assert len(boxes1.shape) == 2 and len(boxes2.shape) == 2 \
           and boxes1.size(1) == 5 and boxes2.size(1) == 5

    # if boxes1.is_cuda:
    #     iou_matrix = _C.rotate_iou_matrix(boxes1, boxes2)
    # else:
    #     iou_matrix = iou_rotate_cpu(boxes1, boxes2)
    #     iou_matrix = torch.FloatTensor(iou_matrix)
    iou_matrix = _C.rotate_iou_matrix(boxes1, boxes2)
    return iou_matrix


def get_rotated_roi_pixel_mapping(roi):
    assert len(roi) == 5  # xc yc w h angle

    xc, yc, w, h, angle = roi

    center = (xc, yc)
    theta = np.deg2rad(angle)

    # paste mask onto image via rotated rect mapping
    v_x = (np.cos(theta), np.sin(theta))
    v_y = (-np.sin(theta), np.cos(theta))
    s_x = center[0] - v_x[0] * ((w - 1) / 2) - v_y[0] * ((h - 1) / 2)
    s_y = center[1] - v_x[1] * ((w - 1) / 2) - v_y[1] * ((h - 1) / 2)

    M = np.array([[v_x[0], v_y[0], s_x],
                  [v_x[1], v_y[1], s_y]])

    return M

def crop_min_area_rect(image, rect):

    '''
    Rotates OpenCV image around center with angle theta (in deg)
    then crops the image according to width and height.
    Taken from https://stackoverflow.com/questions/11627362/how-to-straighten-a-rotated-rectangle-area-of-an-image-using-opencv-in-python/48553593#48553593
    '''
    width = rect[2]
    height = rect[3]

    if np.any(np.isnan(rect)) or np.any(np.isinf(rect)) or width <= 0 or height <= 0:
        return np.zeros((0, 0), dtype=image.dtype)

    mapping = get_rotated_roi_pixel_mapping(rect)

    w = int(np.round(width))
    h = int(np.round(height))
    return cv2.warpAffine(image, mapping, (w, h), flags=cv2.WARP_INVERSE_MAP)


def paste_rotated_roi_in_image(image, roi_image, roi):
    assert len(roi) == 5  # xc yc w h angle

    w = roi[2]
    h = roi[3]

    w = int(np.round(w))
    h = int(np.round(h))
    rh, rw = roi_image.shape[:2]
    if rw != w or rh != h:
        roi_image = cv2.resize(roi_image, (w, h))

    # generate the mapping of points from roi_image to an image
    M = get_rotated_roi_pixel_mapping(roi)

    x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
    x_grid = x_grid.reshape(-1)
    y_grid = y_grid.reshape(-1)
    map_pts_x = x_grid * M[0, 0] + y_grid * M[0, 1] + M[0, 2]
    map_pts_y = x_grid * M[1, 0] + y_grid * M[1, 1] + M[1, 2]
    map_pts_x = np.round(map_pts_x).astype(np.int32)
    map_pts_y = np.round(map_pts_y).astype(np.int32)

    # stick onto image
    im_h, im_w = image.shape[:2]

    valid_x = np.logical_and(map_pts_x >= 0, map_pts_x < im_w)
    valid_y = np.logical_and(map_pts_y >= 0, map_pts_y < im_h)
    valid = np.logical_and(valid_x, valid_y)
    image[map_pts_y[valid], map_pts_x[valid]] = roi_image[y_grid[valid], x_grid[valid]]

    # close holes that arise due to rounding from the pixel mapping phase
    kernel = np.ones((5, 5), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    return image


if __name__ == '__main__':
    import time

    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)

    # =============================NMS ROTATE TEST===================================== #
    from anchor_generator import draw_anchors, convert_rect_to_pts, get_bounding_box, \
        bb_intersection_over_union, draw_bounding_boxes

    def rotate_nms_torch(dets, iou_thresh, device='cpu'):
        nms_rot = RotateNMS(iou_thresh)

        dets_tensor = torch.tensor(dets).to(device)
        keep = nms_rot(dets_tensor)

        return keep.cpu().numpy()


    def standard_nms_cpu(dets, iou_thresh):
        N = dets.shape[0]

        # scores = dets[:,-1]
        # sort_idx = np.argsort(scores)[::-1]
        # dets2 = dets[sort_idx]
        # boxes = dets2[:,:-1]
        boxes = dets

        keep = []
        remv = []
        for r in range(N):
            if r in remv:
                continue

            keep.append(r)
            # r1 = convert_anchor_to_rect(boxes[r])
            # b1 = get_bounding_box(r1)
            b1 = boxes[r]

            for c in range(r + 1, N):
                # r2 = convert_anchor_to_rect(boxes[c])
                # b2 = get_bounding_box(r2)
                b2 = boxes[c]

                iou = bb_intersection_over_union(b1, b2)
                if iou >= iou_thresh:
                    remv.append(c)

        return np.array(keep, dtype=np.uint64)


    dets = np.array([
          [50, 50, 100, 100, 0, 0.99],  # xc,yc,w,h,theta (degrees),score
          [60, 60, 100, 100, 0, 0.88],
          [50, 50, 100, 90, 0., 0.66],
          [50, 50, 100, 100, -45., 0.65],
          [50, 50, 90, 50, 45., 0.6],
          [50, 50, 100, 80, -45., 0.55],
          [150, 150, 200, 30, -45., 0.5],
          [160, 155, 200, 30, -45., 0.46],
          [150, 150, 200, 30, 0., 0.45],
          [170, 170, 200, 30, -45., 0.44],
          [170, 170, 160, 40, 45., 0.435],
          [170, 170, 150, 40, 45., 0.434],
          [170, 170, 150, 42, 45., 0.433],
          [170, 170, 200, 30, 45., 0.43],
          [200, 200, 100, 100, 0., 0.42]
    ], dtype=np.float32)
    dets = np.array([
        [60, 60, 100, 50, -90, 0.9],
        [60, 60, 100, 50, -180, 0.8],
    ], dtype=np.float32)
    dets[dets[:, -2] < -45, -2] += 180
    dets[dets[:, -2] > 135, -2] -= 180

    boxes = dets[:, :-1]
    scores = dets[:, -1]
    rects = np.array([convert_rect_to_pts(b) for b in boxes])
    bounding_boxes = np.array([get_bounding_box(r) for r in rects])

    # dets = np.hstack((boxes, scores))

    iou_thresh = 0.7
    device_id = 0

    device = 'cuda'
    keep = rotate_nms_torch(boxes, iou_thresh, device=device)
    keep2 = nms_rotate_cpu(boxes, iou_thresh, len(boxes))
    print("CPU keep: ", keep2)
    print("GPU keep: ", keep)
    # keep = keep2

    # # =============================IOU ROTATE TEST===================================== #
    #
    # def iou_rotate_torch(boxes1, boxes2, use_gpu=False):
    #
    #     t_boxes1 = torch.FloatTensor(boxes1)
    #     t_boxes2 = torch.FloatTensor(boxes2)
    #     if use_gpu:
    #         t_boxes1 = t_boxes1.cuda()
    #         t_boxes2 = t_boxes2.cuda()
    #
    #     iou_matrix = rotate_iou(t_boxes1, t_boxes2)
    #     iou_matrix = iou_matrix.cpu().numpy()
    #
    #     return iou_matrix
    #
    # boxes1 = np.array([
    #     [50, 50, 100, 300, 0],
    #     [60, 60, 100, 200, 0],
    #     [200, 200, 100, 200, 80.]
    # ], np.float32)
    #
    # boxes2 = np.array([
    #     [50, 50, 100, 300, -45.],
    #     [50, 50, 100, 300, 0.],
    #     [200, 200, 100, 200, 0.],
    #     [200, 200, 100, 200, 90.]
    # ], np.float32)
    #
    # start = time.time()
    # ious = iou_rotate_torch(boxes1, boxes2, use_gpu=False)
    # print(ious)
    # print('{}s'.format(time.time() - start))
    #
    # start = time.time()
    # ious = iou_rotate_torch(boxes1, boxes2, use_gpu=True)
    # print(ious)
    # print('{}s'.format(time.time() - start))
