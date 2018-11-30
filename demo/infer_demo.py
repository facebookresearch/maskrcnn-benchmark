from maskrcnn_benchmark.config import cfg
from predictor import COCODemo, LOVDemo
import numpy as np

import torch
import torch.nn.functional as F

import glob
import cv2


def expand_boxes(boxes, scale):
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    boxes_exp = torch.zeros_like(boxes)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half
    return boxes_exp


def expand_masks(mask, padding):
    N = mask.shape[0]
    M = mask.shape[-1]
    pad2 = 2 * padding
    scale = float(M + pad2) / M
    padded_mask = mask.new_zeros((N, 1, M + pad2, M + pad2))
    padded_mask[:, :, padding:-padding, padding:-padding] = mask
    return padded_mask, scale


def paste_mask_in_image(mask, box, im_h, im_w, padding=1):
    padded_mask, scale = expand_masks(mask[None], padding=padding)
    mask = padded_mask[0, 0]
    box = expand_boxes(box[None], scale)[0]
    box = box.to(dtype=torch.int32)

    TO_REMOVE = 1
    w = box[2] - box[0] + TO_REMOVE
    h = box[3] - box[1] + TO_REMOVE
    w = max(w, 1)
    h = max(h, 1)

    # Set shape to [batchxCxHxW]
    mask = mask.expand((1, 1, -1, -1))

    # Resize mask
    mask = mask.to(torch.float32)
    mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
    mask = mask[0][0]

    # if thresh >= 0:
    #     mask = mask > thresh
    # else:
    #     # for visualization and debugging, we also
    #     # allow it to return an unmodified mask
    #     mask = (mask * 255).to(torch.uint8)

    im_mask = torch.zeros((im_h, im_w), dtype=torch.float32)
    x_0 = max(box[0], 0)
    x_1 = min(box[2] + 1, im_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3] + 1, im_h)

    im_mask[y_0:y_1, x_0:x_1] = mask[
        (y_0 - box[1]) : (y_1 - box[1]), (x_0 - box[0]) : (x_1 - box[0])
    ]
    return im_mask

def normalize(x, xmin=None, xmax=None):
    xmin = np.min(x) if xmin is None else xmin
    xmax = np.max(x) if xmax is None else xmax
    nx = x - xmin
    nx /= (xmax - xmin)
    return nx

def visualize_vertmap(vertmap,bbox,h,w):
    cx = normalize(paste_mask_in_image(vertmap[0, :, :],bbox,h,w).numpy(), -1, 1)
    cy = normalize(paste_mask_in_image(vertmap[1, :, :],bbox,h,w).numpy(), -1, 1)
    cz = normalize(paste_mask_in_image(vertmap[2, :, :],bbox,h,w).numpy(), -1, 1)
    cv2.imshow("vertmap x", cx)
    cv2.imshow("vertmap y", cy)
    cv2.imshow("vertmap z", cz)
    return np.stack((cx,cy,cz), axis=0)

if __name__ == '__main__':

    # config_file = "./configs/mrcnn.yaml"
    config_file = "./configs/lov_debug.yaml"

    # update the config options with the config file
    cfg.merge_from_file(config_file)
    # manual override some options
    # cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

    # coco_demo = COCODemo(
    #     cfg,
    #     min_image_size=480,
    #     confidence_threshold=0.95,
    # )
    lov_demo = LOVDemo(
        cfg,
        min_image_size=480,
        confidence_threshold=0.95,
    )

    # image_dir = "/home/vincent/Documents/py/ml/Detectron.pytorch/demo/coco_debug_images"
    # image_ext = "jpg"
    image_dir = "./datasets/LOV/data/0001"
    image_ext = "1-color.png"
    for image_file in glob.glob("%s/*%s"%(image_dir, image_ext)):
        # load image and then run prediction
        # image_file = "/COCO_val2014_000000010012.jpg"

        img = cv2.imread(image_file)
        if img is None:
            print("Could not find %s"%(image_file))
            continue
        # img = np.expand_dims(img,0)
        res, predictions = lov_demo.run_on_opencv_image(img)
        print("Showing pred results for %s"%(image_file))

        h,w,_ = img.shape

        cv2.imshow("pred", res)
        cv2.waitKey(0)

        verts = predictions.get_field("vertex")#.numpy()
        bboxes = predictions.bbox
        for v, bbox in zip(verts, bboxes):
            vm = visualize_vertmap(v,bbox,h,w)
            # vm = paste_mask_in_image(v[0],bbox,h,w)
            # visualize_vertmap(vm)
            # cv2.imshow("cx", normalize(vm.numpy(), -1, 1))
            cv2.waitKey(0)
