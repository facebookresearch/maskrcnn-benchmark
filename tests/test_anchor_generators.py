import numpy as np
import torch
import cv2

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.rpn.anchor_generator import make_anchor_generator
from maskrcnn_benchmark.modeling.rrpn.anchor_generator import make_anchor_generator as make_rrpn_anchor_generator
from maskrcnn_benchmark.structures.image_list import to_image_list


BLUE = (255,0,0)
RED = (0,0,255)

def convert_rect_to_pts(anchor):
    x_c, y_c, w, h, theta = anchor
    rect = ((x_c, y_c), (w, h), theta)
    rect = cv2.boxPoints(rect)
    # rect = np.int0(np.round(rect))
    return rect

def get_bounding_box(pts):
    """
    pts: (N, 2) array
    """
    bbox = np.zeros(4, dtype=pts.dtype)
    bbox[:2] = np.min(pts, axis=0)
    bbox[2:] = np.max(pts, axis=0)
    return bbox

def get_random_color():
    return (np.random.randint(255), np.random.randint(255), np.random.randint(255))

def draw_anchors(img, anchors, color_list=[], fill=False, line_sz=2):
    """
    img: (H,W,3) np.uint8 array
    anchors: (N,5) np.float32 array, where each row is [xc,yc,w,h,angle]
    """
    if isinstance(color_list, tuple):
        color_list = [color_list]

    img_copy = img.copy()
    Nc = len(color_list)
    N = len(anchors)
    if Nc == 0:
        color_list = [get_random_color() for a in anchors]
    elif Nc != N:
        color_list = [color_list[n % Nc] for n in range(N)]

    for ix,anchor in enumerate(anchors):
        color = color_list[ix]
        rect = anchor
        if len(anchor) != 8:
            rect = convert_rect_to_pts(anchor)
        rect = np.round(rect).astype(np.int32)
        if fill:
            cv2.fillConvexPoly(img_copy, rect, color)
        else:
            cv2.drawContours(img_copy, [rect], 0, color, line_sz)
    return img_copy

def bbox_to_rrect(bb):
    return [(bb[2]+bb[0])/2.0,(bb[3]+bb[1])/2.0,bb[2]-bb[0],bb[3]-bb[1],0]

def get_image_list_and_feature_maps(image, feature_strides):
    N, C, H, W = image.shape
    image_list = to_image_list(image, cfg.DATALOADER.SIZE_DIVISIBILITY)

    feature_maps = [torch.zeros(N,1,H//s,W//s) for s in feature_strides]
    return image_list, feature_maps

def test_rpn(image_tensor):
    USE_FPN = False
    CFG_RPN = cfg.MODEL.RPN
    CFG_RPN.USE_FPN = USE_FPN

    # RPN anchor aspect ratios
    CFG_RPN.ASPECT_RATIOS = (0.5, 1.0, 2.0)
    # Base RPN anchor sizes given in absolute pixels w.r.t. the scaled network input
    CFG_RPN.ANCHOR_SIZES = (32, 64, 128)
    # Stride of the feature map that RPN is attached.
    # For FPN, number of strides should match number of scales
    CFG_RPN.ANCHOR_STRIDE = (16,)
    # Remove RPN anchors that go outside the image by RPN_STRADDLE_THRESH pixels
    # Set to -1 or a large value, e.g. 100000, to disable pruning anchors
    CFG_RPN.STRADDLE_THRESH = 0

    if USE_FPN:
        CFG_RPN.ANCHOR_STRIDE = tuple(np.array(CFG_RPN.ANCHOR_SIZES) // 8)

    image_list, feature_maps = get_image_list_and_feature_maps(image_tensor, CFG_RPN.ANCHOR_STRIDE)

    anchor_generator = make_anchor_generator(cfg)

    print(anchor_generator.num_anchors_per_location())

    all_anchors = anchor_generator.forward(image_list, feature_maps)

    N = len(image_tensor)
    feature_map_id = 0
    for ix in range(N):
        sample_img = np.transpose(image_tensor[ix].numpy(), [1,2,0])
        sample_anchors = all_anchors[ix]
        sample_feature_map_anchors = sample_anchors[feature_map_id]
        anchors_per_stride = anchor_generator.num_anchors_per_location()[feature_map_id]
        bboxes = sample_feature_map_anchors.bbox

        cnt = len(bboxes)

        img_anchors = sample_img.copy().astype(np.uint8) * 255
        for idx in range(0, cnt, anchors_per_stride):
            stride_bboxes = bboxes[idx: idx+anchors_per_stride]
            stride_anchors = [bbox_to_rrect(bb) for bb in stride_bboxes]
            img_anchors2 = draw_anchors(img_anchors, stride_anchors)
            cv2.imshow("anchors", img_anchors2)
            cv2.waitKey(0)

    return all_anchors

def test_rrpn(image_tensor):
    USE_FPN = False

    CFG_RRPN = cfg.MODEL.RPN
    CFG_RRPN.USE_FPN = USE_FPN

    CFG_RRPN.ANCHOR_SIZES = (48, 84, 128)
    # Stride of the feature map that RRPN is attached.
    # For FPN, number of strides should match number of scales
    CFG_RRPN.ANCHOR_STRIDE = (16,)
    # RRPN anchor aspect ratios i.e. width-to-height ratio (RECOMMENDED >= 1.0)
    CFG_RRPN.ASPECT_RATIOS = (1.0, 2.0)
    # RRPN anchor angles (-90 to 90) to represent all possible anchor rotations
    CFG_RRPN.ANCHOR_ANGLES = (-90, -54, -18, 18, 54)  # [-45, -15, 15] #[-90, -75, -60, -45, -30, -15]

    # Remove RRPN anchors that go outside the image by RPN_STRADDLE_THRESH pixels
    # Set to -1 or a large value, e.g. 100000, to disable pruning anchors
    CFG_RRPN.STRADDLE_THRESH = 30

    if USE_FPN:
        CFG_RRPN.ANCHOR_STRIDE = tuple(np.array(CFG_RRPN.ANCHOR_SIZES) // 8)

    image_list, feature_maps = get_image_list_and_feature_maps(image_tensor, CFG_RRPN.ANCHOR_STRIDE)

    anchor_generator = make_rrpn_anchor_generator(cfg)
    all_anchors = anchor_generator.forward(image_list, feature_maps)

    print(anchor_generator.num_anchors_per_location())

    N = len(image_tensor)
    feature_map_id = 0

    for ix in range(N):
        sample_img = np.transpose(image_tensor[ix].numpy(), [1,2,0])
        sample_anchors = all_anchors[ix]
        sample_feature_map_anchors_struct = sample_anchors[feature_map_id]
        sample_feature_map_anchors = sample_feature_map_anchors_struct.get_field("rrects")
        visibility = sample_feature_map_anchors_struct.get_field("visibility")
        anchors_per_stride = anchor_generator.num_anchors_per_location()[feature_map_id]

        cnt = len(sample_feature_map_anchors)

        img_anchors = sample_img.copy().astype(np.uint8) * 255
        for idx in range(0, cnt, anchors_per_stride):
            vis = visibility[idx:idx+anchors_per_stride]
            vis_colors = [RED if v == 0 else BLUE for v in vis]
            stride_anchors = sample_feature_map_anchors[idx:idx+anchors_per_stride]
            img_anchors2 = draw_anchors(img_anchors, stride_anchors, vis_colors)
            cv2.imshow("anchors", img_anchors2)
            cv2.waitKey(0)

    return all_anchors

if __name__ == '__main__':

    N = 1
    C = 3
    H = 240
    W = 320
    image = torch.zeros(N,C,H,W)

    # anchors = test_rpn(image)
    rrpn_anchors = test_rrpn(image)
