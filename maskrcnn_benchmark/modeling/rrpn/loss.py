# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
This file contains specific functions for computing losses on the RPN
file
"""

import torch
from torch.nn import functional as F

# from .utils import concat_box_prediction_layers

from ..balanced_positive_negative_sampler import BalancedPositiveNegativeSampler
# from ..utils import cat

# from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.rotate_ops import rotate_iou
from maskrcnn_benchmark.modeling.rrpn.anchor_generator import convert_rect_to_pts2
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.rrpn.utils import \
    get_boxlist_rotated_rect_tensor, concat_box_prediction_layers

def smooth_l1_loss(input, target, beta=1. / 9):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    return loss

def compute_reg_targets(targets_ori, anchors, box_coder):
    targets = targets_ori.clone()

    dims = targets.shape
    assert len(dims) == 2 and dims[1] == 5 and anchors.shape == dims

    with torch.no_grad():
        if box_coder.relative_angle:
            pass
            # """
            # Get the diff (absolute value) between both angles
            # Normalize the diff to [0, 180]
            # if the angle diff is in range [45, 135]:
            #     flip the target height-width
            #     adjust the target angle -90
            # else if angle diff is in range [135, 180]:
            #     adjust the target angle -180
            # This normalizes the target angle diff to range [-45, 45]
            # """
            # reg_angles = targets[:, -1] - anchors[:, -1]
            # reg_angles_sign = (reg_angles > 0).to(torch.float32)
            # reg_angles_sign[reg_angles_sign == 0] = -1
            # reg_angles_abs = torch.abs(reg_angles)
            #
            # # normalize angle diffs: 0 - 180
            # reg_angles_abs = reg_angles_abs % 180
            #
            # gt_45 = reg_angles_abs > 45
            # gt_135 = reg_angles_abs > 135
            # lt_135 = ~gt_135
            # gt_45_lt_135 = torch.mul(gt_45, lt_135)
            #
            # # if angle diff is in range [45, 135]
            # xd = targets_ori[gt_45_lt_135, 2:4]
            # targets[gt_45_lt_135, 2] = xd[:, 1]
            # targets[gt_45_lt_135, 3] = xd[:, 0]
            #
            # targets[gt_45_lt_135, -1] -= reg_angles_sign[gt_45_lt_135] * 90
            #
            # # if angle diff is in range [135, 180]
            # targets[gt_135, -1] -= reg_angles_sign[gt_135] * 180
        else:
            """
            Normalize anchor angles to range [-45, 45] to match target angles range [-45, 45] 
            """
            pass  # THIS SHOULD BE DONE OUTSIDE AND NOT IN THIS REG TARGET FUNCTION
            # anchors = normalize_anchor_angles(anchors)

    # compute regression targets
    reg_targets = box_coder.encode(
        targets, anchors
    )

    return reg_targets


def trangle_area2(a, b, c):
    return ((a[..., 0] - c[..., 0]) * (b[..., 1] - c[..., 1]) - (a[..., 1] - c[..., 1]) * (b[..., 0] - c[..., 0])) / 2.0


def reorder_pts2(int_pts):
    num_of_inter = len(int_pts)
    if num_of_inter == 0:
        return

    center = torch.mean(int_pts, 0)
    v = int_pts - center
    d = torch.sqrt(v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1])
    v = v / d.unsqueeze(-1)
    v1_lt_0 = v[:, 1] < 0
    v[v1_lt_0, 0] = -2 - v[v1_lt_0, 0]
    vs = v[:, 0]

    # int_pts_f = int_pts.flatten()
    for i in range(num_of_inter):
        if vs[i - 1] > vs[i]:
            temp = vs[i].clone()
            t = int_pts[i].clone()
            j = i
            while (j > 0 and vs[j - 1] > temp):
                vs[j] = vs[j - 1]
                int_pts[j] = int_pts[j - 1]
                j -= 1

            vs[j] = temp
            int_pts[j] = t


def in_rect2(pts1, pts2):
    ab = pts2[:, 1] - pts2[:, 0]
    ab = ab.unsqueeze(-1)
    ad = pts2[:, 3] - pts2[:, 0]
    ad = ad.unsqueeze(-1)
    ap = pts1 - pts2[:, 0].unsqueeze(1)

    abab = ab[:, 0] * ab[:, 0] + ab[:, 1] * ab[:, 1]
    abap = ab[:, 0] * ap[:, :, 0] + ab[:, 1] * ap[:, :, 1]
    adad = ad[:, 0] * ad[:, 0] + ad[:, 1] * ad[:, 1]
    adap = ad[:, 0] * ap[:, :, 0] + ad[:, 1] * ap[:, :, 1]

    return (abab >= abap) * (abap >= 0) * (adad >= adap) * (adap >= 0)


def inter2line2(pts1, pts2):
    device = pts1.device
    mgx, mgy = torch.meshgrid(torch.arange(4), torch.arange(4))
    mgx = mgx.to(device)
    mgy = mgy.to(device)

    a = pts1[:, mgx]  # N,4,4,2
    b = pts1[:, (mgx + 1) % 4]  # N,4,4,2
    c = pts2[:, mgy]  # N,4,4,2
    d = pts2[:, (mgy + 1) % 4]  # N,4,4,2

    area_abc = trangle_area2(a, b, c)
    area_abd = trangle_area2(a, b, d)
    area_cda = trangle_area2(c, d, a)
    area_cdb = area_cda + area_abc - area_abd

    is_intersect = (area_abc * area_abd < 0) * (area_cda * area_cdb < 0)  # N,4,4

    t = area_cda / (area_abd - area_abc)
    dxy = t.unsqueeze(-1) * (b - a)
    intersect_pts = a + dxy  # N,4,4,2

    return is_intersect, intersect_pts


def intersect_area2(int_pts, num_of_inter):
    if num_of_inter <= 2:
        return 0.0

    device = int_pts.device
    a = torch.arange(num_of_inter - 2).to(device)
    area = torch.sum(torch.abs(trangle_area2(int_pts[a * 0], int_pts[a + 1], int_pts[a + 2])))
    return area

def compute_iou_rotate_loss(boxes1, boxes2):
    N = len(boxes1)
    assert N == len(boxes2)
    ious = torch.zeros(N).to(boxes1)

    device = boxes1.device

    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]

    box1_pts = convert_rect_to_pts2(boxes1, lib=torch)
    box2_pts = convert_rect_to_pts2(boxes2, lib=torch)

    b1_in_b2 = in_rect2(box1_pts, box2_pts) # N,4
    b2_in_b1 = in_rect2(box2_pts, box1_pts) # N,4

    is_intersect, intersect_pts = inter2line2(box1_pts, box2_pts)

    for n in range(N):
        int_pts = torch.empty((0, 2), dtype=torch.float32, device=device)
        int_b1 = box1_pts[n][b1_in_b2[n]]
        int_b2 = box2_pts[n][b2_in_b1[n]]
        int_lines = intersect_pts[n][is_intersect[n]]
        for s_pts in [int_b1, int_b2, int_lines]:
            if len(s_pts) > 0:
                int_pts = torch.cat((int_pts, s_pts), 0)

        num_of_inter = int_pts.size(0)
        if num_of_inter > 2:
            reorder_pts2(int_pts)
            int_area = intersect_area2(int_pts, num_of_inter)

            iou = int_area / (area1[n] + area2[n] - int_area)

            ious[n] = iou

    return ious

class RPNLossComputation(object):
    """
    This class computes the RPN loss.
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder):
                 #generate_labels_func):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        # self.target_preparator = target_preparator
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.copied_fields = []
        # self.generate_labels_func = generate_labels_func
        self.discard_cases = ['not_visibility', 'between_thresholds']

    def match_targets_to_anchors(self, anchor, target, copied_fields=[]):
        """
        :param anchor: BoxList
        :param target: BoxList
        :param copied_fields: list
        :return:
        """
        masks_field = "masks"
        rrects_field = "rrects"

        # anchor_tensor = anchor.get_field(rrects_field)
        anchor_tensor = get_boxlist_rotated_rect_tensor(anchor, masks_field, rrects_field)
        target_tensor = get_boxlist_rotated_rect_tensor(target, masks_field, rrects_field)
        if not target.has_field(rrects_field):  # add rrects to gt if not already added
            target.add_field(rrects_field, target_tensor)

        match_quality_matrix = rotate_iou(target_tensor, anchor_tensor)

        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # RPN doesn't need any fields from target
        # for creating the labels, so clear them all
        target = target.copy_with_fields(copied_fields + [rrects_field])
        # get the targets corresponding GT for each anchor
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_idxs_clamped = matched_idxs.clamp(min=0)
        matched_targets = target[matched_idxs_clamped]
        # matched_ious = match_quality_matrix[matched_idxs_clamped,
        #                 torch.arange(len(anchor_tensor), device=anchor_tensor.device)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        # matched_targets.add_field("matched_ious", matched_ious)
        return matched_targets

    def prepare_targets(self, anchors, targets):
        labels = []
        regression_targets = []
        matched_gt_ids_per_image = []
        matched_gt_ious_per_image = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            matched_targets = self.match_targets_to_anchors(
                anchors_per_image, targets_per_image, self.copied_fields
            )

            matched_idxs = matched_targets.get_field("matched_idxs")
            labels_per_image = matched_idxs >= 0
            labels_per_image = labels_per_image.to(dtype=torch.float32)

            # Background (negative examples)
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0

            # discard anchors that go out of the boundaries of the image
            if "not_visibility" in self.discard_cases:
                labels_per_image[~anchors_per_image.get_field("visibility")] = -1

            # discard indices that are between thresholds
            if "between_thresholds" in self.discard_cases:
                inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1

            target_rrects = matched_targets.get_field("rrects")
            anchor_rrects = anchors_per_image.get_field("rrects")

            # compute regression targets
            regression_targets_per_image = compute_reg_targets(
                target_rrects, anchor_rrects, self.box_coder
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)
            matched_gt_ids_per_image.append(matched_idxs)
            # matched_gt_ious_per_image.append(matched_targets.get_field("matched_ious"))

        return labels, regression_targets, matched_gt_ids_per_image, matched_gt_ious_per_image


    def __call__(self, anchors, objectness, box_regression, targets):
        """
        Arguments:
            anchors (list[BoxList])
            objectness (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor
        """

        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]

        labels, regression_targets, matched_gt_ids, \
            matched_gt_ious = self.prepare_targets(anchors, targets)

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        labels = torch.cat(labels, dim=0)

        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        total_pos = sampled_pos_inds.numel()
        total_neg = sampled_neg_inds.numel()
        total_samples = total_pos + total_neg

        objectness, box_regression = concat_box_prediction_layers(objectness, box_regression)
        objectness = objectness.squeeze()

        if total_pos == 0:
            return objectness.sum() * 0, objectness.sum() * 0

        regression_targets = torch.cat(regression_targets, dim=0)

        with torch.no_grad():
            start_gt_idx = 0
            for ix, t in enumerate(targets):
                matched_gt_ids[ix] += start_gt_idx
                start_gt_idx += len(t)

            matched_gt_ids = torch.cat(matched_gt_ids)
            pos_matched_gt_ids = matched_gt_ids[sampled_pos_inds]

            pos_label_weights = torch.zeros_like(pos_matched_gt_ids, dtype=torch.float32)

            label_idxs = [torch.nonzero(pos_matched_gt_ids == x).squeeze() for x in range(start_gt_idx)]

            # """OLD"""
            label_cnts = [li.numel() for li in label_idxs]
            # label_weights = total_pos / label_cnts.to(dtype=torch.float32)
            # label_weights /= start_gt_idx  # equal class weighting
            for x in range(start_gt_idx):
                if label_cnts[x] > 0:
                    pos_label_weights[label_idxs[x]] = total_pos / label_cnts[x] / start_gt_idx  # equal class weighting
        #
        #     # # """NEW"""
        #     # MAX_GT_NUM = 6  # TODO: CONFIG
        #     # matched_gt_ious = torch.cat(matched_gt_ious)
        #     # pos_matched_gt_ious = matched_gt_ious[sampled_pos_inds]
        #     #
        #     # label_cnts = [min(MAX_GT_NUM, nz.numel()) for nz in label_idxs]
        #     # total_pos = sum(label_cnts)
        #     # for x in range(start_gt_idx):
        #     #     nz = label_idxs[x]
        #     #     nnn = nz.numel()
        #     #     if nnn <= MAX_GT_NUM:
        #     #         if nnn > 0:
        #     #             pos_label_weights[nz] = total_pos / nnn
        #     #         continue
        #     #     top_iou_ids = torch.sort(pos_matched_gt_ious[nz], descending=True)[1][:MAX_GT_NUM]
        #     #     inds = nz[top_iou_ids]
        #     #     pos_label_weights[inds] = total_pos / MAX_GT_NUM
        #     #
        #     # pos_label_weights = pos_label_weights / start_gt_idx
        #
        # pos_regression = box_regression[sampled_pos_inds]
        # pos_regression_targets = regression_targets[sampled_pos_inds]
        # # normalize_reg_targets(pos_regression_targets)
        # box_loss = smooth_l1_loss(
        #     pos_regression,#[:, :-1],
        #     pos_regression_targets,#[:, :-1],
        #     beta=1.0 / 9,
        # )
        # box_loss = (box_loss * pos_label_weights.unsqueeze(1)).sum() / total_pos
        #
        # # angle_loss = 0 #torch.abs(torch.sin(pos_regression[:, -1] - pos_regression_targets[:, -1])).mean()
        #
        # # balance negative and positive weights
        sampled_labels = labels[sampled_inds]
        objectness_weights = torch.ones_like(sampled_labels, dtype=torch.float32)
        objectness_weights[sampled_labels == 1] = pos_label_weights
        objectness_weights[sampled_labels != 1] = min(pos_label_weights.min(), 0.5)

        # criterion = torch.nn.BCELoss(reduce=False)
        # entropy_loss = criterion(objectness[sampled_inds].sigmoid(), sampled_labels)
        # objectness_loss = torch.mul(entropy_loss, objectness_weights).sum()

        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], sampled_labels, weight=objectness_weights
        )

        box_reg = box_regression[sampled_pos_inds]
        box_reg_targets = regression_targets[sampled_pos_inds]
        # box_loss = smooth_l1_loss(
        #     box_reg,
        #     box_reg_targets,
        #     beta=1.0 / 9,
        #     # size_average=False,
        # ).sum() / (total_samples)

        base_anchors = torch.cat([a.get_field("rrects") for a in anchors])[sampled_pos_inds]
        pred_box = self.box_coder.decode(box_reg, base_anchors)
        gt_box = self.box_coder.decode(box_reg_targets, base_anchors)
        ious = compute_iou_rotate_loss(pred_box, gt_box)
        iou_loss = torch.where(ious <= 0, ious * 0.0, -torch.log(ious ** 2))
        box_loss = iou_loss.sum() / total_samples

        # objectness_loss = F.binary_cross_entropy_with_logits(
        #     objectness[sampled_inds], labels[sampled_inds]
        # )
        return objectness_loss, box_loss#, angle_loss


def make_rpn_loss_evaluator(cfg, box_coder):
    matcher = Matcher(
        cfg.MODEL.RPN.FG_IOU_THRESHOLD,
        cfg.MODEL.RPN.BG_IOU_THRESHOLD,
        allow_low_quality_matches=True,
    )

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION
    )

    loss_evaluator = RPNLossComputation(
        matcher,
        fg_bg_sampler,
        box_coder,
        # generate_rpn_labels
    )
    return loss_evaluator
