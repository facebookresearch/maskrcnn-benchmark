# Mask Scoring R-CNN
# Wriiten by zhaojin.huang, 2018-12.

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from maskrcnn_benchmark.structures.bounding_box import BoxList


# TODO get the predicted maskiou and mask score.
class MaskIoUPostProcessor(nn.Module):
    """
    Getting the maskiou according to the targeted label, and computing the mask score according to maskiou.
    """

    def __init__(self):
        super(MaskIoUPostProcessor, self).__init__()

    def forward(self, boxes, pred_maskiou, labels):
        for box, pm_iou, label in zip(boxes, pred_maskiou, labels):
            num_masks = pm_iou.shape[0]
            index = torch.arange(num_masks, device=label.device)
            maskious = pm_iou[index, label]

            bbox_scores = box.get_field("scores")
            mask_scores = bbox_scores * maskious
            box.add_field("mask_scores", mask_scores)

        return boxes

def make_roi_maskiou_post_processor(cfg):
    maskiou_post_processor = MaskIoUPostProcessor()
    return maskiou_post_processor
