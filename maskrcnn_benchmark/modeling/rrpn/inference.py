# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from maskrcnn_benchmark.modeling.rotated_box_coder import BoxCoder
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.rrpn.anchor_generator import convert_rects_to_bboxes
from maskrcnn_benchmark.modeling.rrpn.utils import get_boxlist_rotated_rect_tensor
from maskrcnn_benchmark.modeling.rotate_ops import RotateNMS
# from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
# from maskrcnn_benchmark.structures.boxlist_ops import remove_small_boxes

from .utils import REGRESSION_CN, permute_and_flatten

def remove_small_boxes(proposal, min_size):
    """
    Arguments:
        proposal (tensor): (N, 5)  xc,yc,w,h,theta
        min_size (int)
    """
    _, _, ws, hs, _ = proposal.unbind(dim=1)
    keep = (
        (ws >= min_size) & (hs >= min_size)
    ).nonzero().squeeze(1)
    return keep


class RPNPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RPN boxes, before feeding the
    proposals to the heads
    """

    def __init__(
        self,
        box_coder,
        pre_nms_top_n,
        post_nms_top_n,
        nms_thresh,
        min_size,
        fpn_post_nms_top_n=None,
    ):
        """
        Arguments:
            pre_nms_top_n (int)
            post_nms_top_n (int)
            nms_thresh (float)
            min_size (int)
            box_coder (BoxCoder)
            fpn_post_nms_top_n (int)
        """
        super(RPNPostProcessor, self).__init__()
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size

        self.box_coder = box_coder

        if fpn_post_nms_top_n is None:
            fpn_post_nms_top_n = post_nms_top_n
        self.fpn_post_nms_top_n = fpn_post_nms_top_n

        self.nms_rotate = RotateNMS(nms_threshold=nms_thresh, post_nms_top_n=post_nms_top_n)

    def add_gt_proposals(self, proposals, targets):
        """
        Arguments:
            proposals: list[BoxList]
            targets: list[BoxList]
        """
        # Get the device we're operating on
        device = proposals[0].bbox.device

        masks_field = "masks"
        rrects_field = "rrects"

        gt_boxes = [target.copy_with_fields([]) for target in targets]

        # later cat of bbox requires all fields to be present for all bbox
        # so we need to add a dummy for objectness that's missing
        for ix,gt_box in enumerate(gt_boxes):

            target = targets[ix]
            gt_rrect = get_boxlist_rotated_rect_tensor(target, masks_field, rrects_field)

            aug_rrect = gt_rrect.clone()
            N = len(gt_rrect)

            multiplier = int(4)
            for m in range(multiplier-1):
                aug_rrect[:, :2] += torch.randint(-7, 7, (N, 2), dtype=torch.float32, device=device)
                aug_rrect[:, 2:4] *= torch.randint(94, 106, (N, 2), dtype=torch.float32, device=device) / 100
                aug_rrect[:, -1] += torch.randint(-5, 5, (N,), dtype=torch.float32, device=device)
                gt_rrect = torch.cat((gt_rrect, aug_rrect))

            # convert anchor rects to bboxes
            bboxes = convert_rects_to_bboxes(gt_rrect, torch)

            boxlist = BoxList(bboxes, gt_box.size, mode="xyxy")

            boxlist.add_field("rrects", gt_rrect)
            boxlist.add_field("objectness", torch.ones(len(gt_rrect), device=device))
            gt_boxes[ix] = boxlist

        proposals = [
            cat_boxlist((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def forward_for_single_feature_map(self, anchors, objectness, box_regression):
        """
        Arguments:
            anchors: list[BoxList]
            objectness: tensor of size N, A, H, W
            box_regression: tensor of size N, A * REGRESSION_CN, H, W
        """
        device = objectness.device
        N, A, H, W = objectness.shape

        # put in the same format as anchors
        objectness = permute_and_flatten(objectness, N, A, 1, H, W).view(N, -1)
        objectness = objectness.sigmoid()

        box_regression = permute_and_flatten(box_regression, N, A, REGRESSION_CN, H, W)

        num_anchors = A * H * W

        pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)
        objectness, topk_idx = objectness.topk(pre_nms_top_n, dim=1, sorted=True)  # sorted!

        batch_idx = torch.arange(N, device=device)[:, None]
        box_regression = box_regression[batch_idx, topk_idx]

        image_shapes = [box.size for box in anchors]
        concat_anchors = torch.cat([a.get_field("rrects") for a in anchors], dim=0)
        concat_anchors = concat_anchors.reshape(N, -1, REGRESSION_CN)[batch_idx, topk_idx]

        proposals = self.box_coder.decode(
            box_regression.view(-1, REGRESSION_CN), concat_anchors.view(-1, REGRESSION_CN)
        )

        proposals = proposals.view(N, -1, REGRESSION_CN)

        result = []
        for proposal, score, im_shape in zip(proposals, objectness, image_shapes):
            # filter small boxes
            if self.min_size > 0:
                keep = remove_small_boxes(proposal, self.min_size)
                proposal = proposal[keep]
                score = score[keep]

            # perform rotated nms (no need to sort, since already sorted above)
            keep = self.nms_rotate(proposal)
            proposal = proposal[keep]
            score = score[keep]

            # convert anchor rects to bboxes
            bboxes = convert_rects_to_bboxes(proposal, torch)

            boxlist = BoxList(bboxes, im_shape, mode="xyxy")

            boxlist.add_field("rrects", proposal)
            boxlist.add_field("objectness", score)

            boxlist = boxlist.clip_to_image(remove_empty=False)
            result.append(boxlist)
        return result

    def forward(self, anchors, objectness, box_regression, targets=None):
        """
        Arguments:
            anchors: list[list[BoxList]]
            objectness: list[tensor]
            box_regression: list[tensor]

        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = []
        num_levels = len(objectness)
        anchors = list(zip(*anchors))
        for a, o, b in zip(anchors, objectness, box_regression):
            sampled_boxes.append(self.forward_for_single_feature_map(a, o, b))

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]

        if num_levels > 1:
            boxlists = self.select_over_all_levels(boxlists)

        # append ground-truth bboxes to proposals
        if self.training and targets is not None:
            boxlists = self.add_gt_proposals(boxlists, targets)

        return boxlists

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        # different behavior during training and during testing:
        # during training, post_nms_top_n is over *all* the proposals combined, while
        # during testing, it is over the proposals for each image
        # TODO resolve this difference and make it consistent. It should be per image,
        # and not per batch
        if self.training:
            objectness = torch.cat(
                [boxlist.get_field("objectness") for boxlist in boxlists], dim=0
            )
            box_sizes = [len(boxlist) for boxlist in boxlists]
            post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
            _, inds_sorted = torch.topk(objectness, post_nms_top_n, dim=0, sorted=True)
            inds_mask = torch.zeros_like(objectness, dtype=torch.uint8)
            inds_mask[inds_sorted] = 1
            inds_mask = inds_mask.split(box_sizes)
            for i in range(num_images):
                boxlists[i] = boxlists[i][inds_mask[i]]
        else:
            for i in range(num_images):
                objectness = boxlists[i].get_field("objectness")
                post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
                _, inds_sorted = torch.topk(
                    objectness, post_nms_top_n, dim=0, sorted=True
                )
                boxlists[i] = boxlists[i][inds_sorted]
        return boxlists


def make_rpn_postprocessor(config, rpn_box_coder, is_train):
    CFG_RPN = config.MODEL.RPN

    fpn_post_nms_top_n = CFG_RPN.FPN_POST_NMS_TOP_N_TRAIN
    if not is_train:
        fpn_post_nms_top_n = CFG_RPN.FPN_POST_NMS_TOP_N_TEST

    pre_nms_top_n = CFG_RPN.PRE_NMS_TOP_N_TRAIN
    post_nms_top_n = CFG_RPN.POST_NMS_TOP_N_TRAIN
    if not is_train:
        pre_nms_top_n = CFG_RPN.PRE_NMS_TOP_N_TEST
        post_nms_top_n = CFG_RPN.POST_NMS_TOP_N_TEST
    nms_thresh = CFG_RPN.NMS_THRESH
    min_size = CFG_RPN.MIN_SIZE
    box_selector = RPNPostProcessor(
        pre_nms_top_n=pre_nms_top_n,
        post_nms_top_n=post_nms_top_n,
        nms_thresh=nms_thresh,
        min_size=min_size,
        box_coder=rpn_box_coder,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
    )
    return box_selector
