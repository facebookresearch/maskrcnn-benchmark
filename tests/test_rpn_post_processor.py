import numpy as np
import torch
# import cv2

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList

from maskrcnn_benchmark.modeling.rrpn.anchor_generator import \
    make_anchor_generator as make_rrpn_anchor_generator, convert_rect_to_pts2
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask


def get_image_list_and_feature_maps(image, feature_strides):
    N, C, H, W = image.shape
    image_list = to_image_list(image, cfg.DATALOADER.SIZE_DIVISIBILITY)

    feature_maps = [torch.zeros(N,1,H//s,W//s, device=image.device) for s in feature_strides]
    return image_list, feature_maps



def test_rpn_post_processor(image_tensor, targets_data):
    from maskrcnn_benchmark.modeling.rpn.anchor_generator import make_anchor_generator
    from maskrcnn_benchmark.modeling.rpn.inference import make_rpn_postprocessor
    from maskrcnn_benchmark.modeling.rpn.loss import make_rpn_loss_evaluator

    N, C, H, W = image_tensor.shape
    targets = [BoxList(td, (W,H), mode="xyxy") for td in targets_data]

    device = image_tensor.device

    REGRESSION_CN = 4

    USE_FPN = False
    CFG_RPN = cfg.MODEL.RPN

    if USE_FPN:
        CFG_RPN.ANCHOR_STRIDE = tuple(np.array(CFG_RPN.ANCHOR_SIZES) // 8)

    image_list, feature_maps = get_image_list_and_feature_maps(image_tensor, CFG_RPN.ANCHOR_STRIDE)

    anchor_generator = make_anchor_generator(cfg)
    num_anchors = anchor_generator.num_anchors_per_location()

    anchors = anchor_generator.forward(image_list, feature_maps)

    objectness = []
    box_regression = []
    for ix, fm in enumerate(feature_maps):
        n_anchors = num_anchors[ix]
        N,_,h,w = fm.shape
        objectness.append(torch.rand(N, n_anchors, h, w, device=device))
        box_regression.append(torch.rand(N, n_anchors*REGRESSION_CN, h, w, device=device))

    # train mode
    postprocessor_train = make_rpn_postprocessor(cfg, rpn_box_coder=None, is_train=True)
    postprocessor_train.train()

    result = postprocessor_train.forward(anchors, objectness, box_regression, targets=targets)

    # check loss
    loss_evaluator = make_rpn_loss_evaluator(cfg, postprocessor_train.box_coder)
    loss_objectness, loss_rpn_box_reg = loss_evaluator(anchors, objectness, box_regression, targets)

    # test mode
    postprocessor_test = make_rpn_postprocessor(cfg, rpn_box_coder=None, is_train=False)
    postprocessor_test.eval()

    result = postprocessor_test.forward(anchors, objectness, box_regression)


def test_rrpn_post_processor(image_tensor, targets_data):
    from maskrcnn_benchmark.modeling.rrpn.inference import make_rpn_postprocessor, REGRESSION_CN
    from maskrcnn_benchmark.modeling.rrpn.loss import make_rpn_loss_evaluator

    N, C, H, W = image_tensor.shape


    targets = []
    for ix, td in enumerate(targets_data):
        rect_pts = convert_rect_to_pts2(td)#.reshape((len(td), 8))
        nn = len(rect_pts)
        bboxes = np.zeros((nn, 4), dtype=np.float32)
        bboxes[:, :2] = np.min(rect_pts, axis=1)
        bboxes[:, 2:] = np.max(rect_pts, axis=1)
        boxlist = BoxList(bboxes, (W,H), mode="xyxy")
        mm = SegmentationMask(rect_pts.reshape(nn, 1, 8).tolist(), (W,H), mode='poly')
        boxlist.add_field("masks", mm)
        targets.append(boxlist)

    device = image_tensor.device


    USE_FPN = False
    cfg.MODEL.ROTATED = True
    CFG_RPN = cfg.MODEL.RPN

    CFG_RPN.ANCHOR_ANGLES = (-90, -54, -18, 18, 54)
    CFG_RPN.ANCHOR_SIZES = (48, 84, 128, 224)
    CFG_RPN.ANCHOR_STRIDE = (16,)
    CFG_RPN.ASPECT_RATIOS = (1.0, 2.0)

    if USE_FPN:
        CFG_RPN.ANCHOR_STRIDE = tuple(np.array(CFG_RPN.ANCHOR_SIZES) // 8)
    CFG_RPN.POST_NMS_TOP_N_TRAIN = 100

    image_list, feature_maps = get_image_list_and_feature_maps(image_tensor, CFG_RPN.ANCHOR_STRIDE)

    anchor_generator = make_rrpn_anchor_generator(cfg)
    num_anchors = anchor_generator.num_anchors_per_location()

    anchors = anchor_generator.forward(image_list, feature_maps)

    objectness = []
    box_regression = []
    for ix, fm in enumerate(feature_maps):
        n_anchors = num_anchors[ix]
        N,_,h,w = fm.shape
        objectness.append(torch.rand(N, n_anchors, h, w, device=device))
        box_regression.append(torch.rand(N, n_anchors*REGRESSION_CN, h, w, device=device))

    # train mode
    postprocessor_train = make_rpn_postprocessor(cfg, rpn_box_coder=None, is_train=True)
    postprocessor_train.train()

    # result = postprocessor_train.forward(anchors, objectness, box_regression, targets=targets)

    # check loss
    loss_evaluator = make_rpn_loss_evaluator(cfg, postprocessor_train.box_coder)
    loss_objectness, loss_rpn_box_reg = loss_evaluator(anchors, objectness, box_regression, targets)

    # test mode
    postprocessor_test = make_rpn_postprocessor(cfg, rpn_box_coder=None, is_train=False)
    postprocessor_test.eval()

    result = postprocessor_test.forward(anchors, objectness, box_regression)



if __name__ == '__main__':
    N = 1
    C = 3
    H = 160
    W = 240

    device = 'cpu'
    image = torch.zeros(N,C,H,W, device=device)
    targets = np.array([
        [50, 50, 100, 100, 0],
        [50, 50, 50, 50, -90]
    ], dtype=np.float32)
    bbox_targets = np.array([
        [0, 0, 100, 100],
        [25, 25, 75, 75]
    ], dtype=np.float32)

    targets = [targets for ix in range(N)]
    bbox_targets = [bbox_targets for ix in range(N)]

    test_rpn_post_processor(image, bbox_targets)
    test_rrpn_post_processor(image, targets)

    from maskrcnn_benchmark.modeling.rrpn.utils import get_segmentation_mask_rotated_rect_tensor
    tt = []
    for ix, td in enumerate(targets):
        rect_pts = convert_rect_to_pts2(td)#.reshape((len(td), 8))
        nn = len(rect_pts)
        bboxes = np.zeros((nn, 4), dtype=np.float32)
        bboxes[:, :2] = np.min(rect_pts, axis=1)
        bboxes[:, 2:] = np.max(rect_pts, axis=1)
        boxlist = BoxList(bboxes, (W,H), mode="xyxy")
        mm = SegmentationMask(rect_pts.reshape(nn, 1, 8).tolist(), (W,H), mode='poly')
        boxlist.add_field("masks", mm)
        tt.append(boxlist)

        rrect_tensor = get_segmentation_mask_rotated_rect_tensor(mm)
