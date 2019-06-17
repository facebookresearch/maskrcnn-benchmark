import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# Heuristic choice based on that would scale a 16 pixel anchor up to 1000 pixels
BBOX_XFORM_CLIP = np.log(1000. / 16.)

class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights, bbox_xform_clip=BBOX_XFORM_CLIP):
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """

        TO_REMOVE = 1  # TODO remove
        ex_widths = proposals[:, 2] - proposals[:, 0] + TO_REMOVE
        ex_heights = proposals[:, 3] - proposals[:, 1] + TO_REMOVE
        ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
        ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights

        gt_widths = reference_boxes[:, 2] - reference_boxes[:, 0] + TO_REMOVE
        gt_heights = reference_boxes[:, 3] - reference_boxes[:, 1] + TO_REMOVE
        gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
        gt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_heights

        wx, wy, ww, wh = self.weights
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)

        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
        return targets

    def decode(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """

        boxes = boxes.to(rel_codes.dtype)

        TO_REMOVE = 1  # TODO remove
        widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
        heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(rel_codes)
        # x1
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        # y1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
        # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

        return pred_boxes


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
 
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    # return the intersection over union value
    return iou

def compute_iou_loss(pred, target, base_box, box_coder, batch_size=None):
    """
    Adapted from https://github.com/generalized-iou/Detectron.pytorch/blob/master/lib/utils/net.py
    """
    if batch_size is None:
        batch_size = pred.size(0)

    pred_box = box_coder.decode(pred, base_box)
    gt_box = box_coder.decode(target, base_box)

    x1, y1, x2, y2 = pred_box.split(1, dim=1)
    x1g, y1g, x2g, y2g = gt_box.split(1, dim=1)

    x2 = torch.max(x1, x2)
    y2 = torch.max(y1, y2)

    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    intsctk = torch.zeros(x1.size()).to(pred)
    mask = (ykis2 > ykis1) * (xkis2 > xkis1)
    intsctk[mask] = ((xkis2 - xkis1) * (ykis2 - ykis1))[mask]
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk + 1e-7
    iouk = intsctk / unionk

    area_c = (xc2 - xc1) * (yc2 - yc1) + 1e-7
    miouk = iouk - ((area_c - unionk) / area_c)
    iou_weights = 1.0 # bbox_inside_weights.view(-1, 4).mean(1) * bbox_outside_weights.view(-1, 4).mean(1)
    # iouk = ((1 - iouk) * iou_weights).sum(0) / batch_size
    iouk = ((-torch.log(iouk)) * iou_weights).sum(0) / batch_size
    # miouk = ((1 - miouk) * iou_weights).sum(0) / batch_size
    miouk = ((1 - miouk) * iou_weights).sum(0) / batch_size

    return iouk, miouk


def smooth_l1_loss(input, target, beta=1. / 9, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()


class SimpleLinearModel(nn.Module):
    def __init__(self, regression_cn=4):
        super(SimpleLinearModel, self).__init__()

        hidden_layer = 5
        self.fc_layer1 = nn.Linear(regression_cn, hidden_layer)
        self.fc_layer2 = nn.Linear(regression_cn, hidden_layer)
        self.fc_out_layer = nn.Linear(hidden_layer*2, 16)
        self.fc_out_layer2 = nn.Linear(16, regression_cn)
        self.bn = nn.BatchNorm1d(hidden_layer*2)

        for layer in [self.fc_layer1, self.fc_layer2, self.fc_out_layer, self.fc_out_layer2]:
            nn.init.normal_(layer.weight, mean=0, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, input1, input2):
        out1 = self.fc_layer1(input1)
        out2 = self.fc_layer2(input2)
        x = torch.cat((out1, out2), 1)
        x = self.bn(F.relu(x))
        x = F.relu(self.fc_out_layer(x))
        out = self.fc_out_layer2(x)
        return out

if __name__ == '__main__':
    anchor = np.array([
        [80,80,160,160],
        [90,90,105,105],
        [50,50,100,100],
    ], dtype=np.float32)
    bbox_gt = np.array([
        [100,100,190,190], # xyxy
        [100,100,120,120],
        [40,40,80,80],
    ], dtype=np.float32)

    N = len(bbox_gt)

    t_bbox_gt = torch.from_numpy(bbox_gt)
    t_anchor = torch.from_numpy(anchor)

    box_coder = BoxCoder(weights=(1.0,1.0,1.0,1.0))
    regression_targets = box_coder.encode(
        t_bbox_gt, t_anchor
    )

    model = SimpleLinearModel(regression_cn=4)

    import torch.optim as optim
    lr = 3e-3
    n_iters = 100

    optimizer = optim.Adam(model.parameters(), lr=lr)#, betas=(0.9, 0.999))
    for n in range(n_iters):
        
        regression_pred = model((t_anchor-100)/200, (t_bbox_gt-100)/200)

        # l1_loss = smooth_l1_loss(regression_pred, regression_targets, beta=1.0/9)
        iou_loss, giou_loss = compute_iou_loss(regression_pred, regression_targets, t_anchor, box_coder)

        loss = iou_loss
        # print("Loss: %.3f"%loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            bbox_pred = box_coder.decode(
                regression_pred.view(-1, 4), t_anchor.view(-1, 4)
            ).numpy()
            mean_iou = np.mean([bb_intersection_over_union(bbox_pred[n], bbox_gt[n]) for n in range(N)])
            print("Iter %d) Loss: %.3f, Mean IoU: %.2f"%(n, loss.item(), mean_iou))
