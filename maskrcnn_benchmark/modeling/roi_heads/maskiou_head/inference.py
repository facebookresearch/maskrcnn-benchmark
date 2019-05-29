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
        num_masks = pred_maskiou.shape[0]
        index = torch.arange(num_masks, device=labels.device)
        maskious = pred_maskiou[index, labels]
        maskious = [maskious]
        results = []
        for maskiou, box in zip(maskious, boxes):
            bbox = BoxList(box.bbox, box.size, mode="xyxy")
            for field in box.fields():
                bbox.add_field(field, box.get_field(field))
            bbox_scores = bbox.get_field("scores")
            mask_scores = bbox_scores * maskiou
            bbox.add_field("mask_scores", mask_scores)
            results.append(bbox)

        return results

def make_roi_maskiou_post_processor(cfg):
    maskiou_post_processor = MaskIoUPostProcessor()
    return maskiou_post_processor
