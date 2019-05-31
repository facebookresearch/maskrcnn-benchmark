# Mask Scoring R-CNN
# Wriiten by zhaojin.huang, 2018-12.

import torch
from torch.nn import functional as F

from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat

def l2_loss(input, target):
    cond = torch.abs(input - target)
    loss = 0.5 * cond**2 / input.shape[0]
    return loss.sum()


class MaskIoULossComputation(object):
    def __init__(self, loss_weight):
        self.loss_weight = loss_weight

    def __call__(self, labels, pred_maskiou, gt_maskiou):

        positive_inds = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[positive_inds]
        if labels_pos.numel() == 0:
            return pred_maskiou.sum() * 0
        # gt_maskiou = gt_maskiou.detach()

        gt_miou = gt_maskiou.detach()
        pos_inds = torch.nonzero(gt_miou > 0.0).squeeze(1)
        if pos_inds.numel() == 0:
            return pred_maskiou.sum() * 0
        maskiou_loss = l2_loss(pred_maskiou[positive_inds, labels_pos][pos_inds], gt_miou[pos_inds])
        maskiou_loss = self.loss_weight * maskiou_loss

        return maskiou_loss


def make_roi_maskiou_loss_evaluator(cfg):
    loss_weight = cfg.MODEL.ROI_MASKIOU_HEAD.LOSS_WEIGHT
    loss_evaluator = MaskIoULossComputation(loss_weight)

    return loss_evaluator
