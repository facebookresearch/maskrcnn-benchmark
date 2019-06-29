# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.modeling.rotated_box_coder import BoxCoder
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.rrpn.anchor_generator import convert_rects_to_bboxes
# from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.layers.rotate_nms import RotateNMS

REGRESSION_CN = 5

class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self,
        box_coder,
        score_thresh=0.05,
        nms=0.5,
        detections_per_img=100,
        cls_agnostic_bbox_reg=False,
        use_nms=True
    ):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
        """
        super(PostProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

        self.nms_rotate = RotateNMS(nms_threshold=nms)#, post_nms_top_n=-1)
        self.use_nms = use_nms
        if not self.use_nms:
            self.detections_per_img = -1

    def forward(self, x, boxes):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        class_logits, box_regression = x
        class_prob = F.softmax(class_logits, -1)

        # TODO think about a representation of batch of boxes
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.get_field("rrects") for a in boxes], dim=0)

        if self.cls_agnostic_bbox_reg:
            box_regression = box_regression[:, -REGRESSION_CN:]

        box_regression = box_regression.view(sum(boxes_per_image), -1)
        # box_regression[:] = 0
        proposals = self.box_coder.decode(
            box_regression, concat_boxes
        )
        if self.cls_agnostic_bbox_reg:
            proposals = proposals.repeat(1, class_prob.shape[1])

        num_classes = class_prob.shape[1]

        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)

        results = []
        for prob, proposals_per_img, image_shape in zip(
            class_prob, proposals, image_shapes
        ):
            boxlist = self.prepare_boxlist(proposals_per_img, prob, image_shape)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = self.filter_results(boxlist, num_classes)
            results.append(boxlist)
        return results

    def prepare_boxlist(self, proposals, scores, image_shape):
        """
        Returns BoxList from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, REGRESSION_CN * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * REGRESSION_CN:(j + 1) * REGRESSION_CN]`.
        """
        proposals = proposals.reshape(-1, REGRESSION_CN)
        scores = scores.reshape(-1)

        # convert anchor rects to bboxes
        bboxes = convert_rects_to_bboxes(proposals, torch)

        boxlist = BoxList(bboxes, image_shape, mode="xyxy")

        boxlist.add_field("rrects", proposals)
        boxlist.add_field("scores", scores)

        return boxlist

    def filter_results(self, boxlist, num_classes):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        bboxes = boxlist.bbox.reshape(-1, num_classes * 4)
        rrects = boxlist.get_field("rrects").reshape(-1, num_classes * REGRESSION_CN)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)

        device = scores.device
        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = scores > self.score_thresh
        for j in range(1, num_classes):
            inds = inds_all[:, j].nonzero().squeeze(1)
            bboxes_j = bboxes[inds, j * 4: (j + 1) * 4]
            rrects_j = rrects[inds, j * REGRESSION_CN : (j + 1) * REGRESSION_CN]
            scores_j = scores[inds, j]

            # sort scores!
            sorted_idx = torch.sort(scores_j, descending=True)[1]
            bboxes_j = bboxes_j[sorted_idx]
            rrects_j = rrects_j[sorted_idx]
            scores_j = scores_j[sorted_idx]

            # perform nms
            if self.use_nms:
                keep = self.nms_rotate(rrects_j)
                bboxes_j = bboxes_j[keep]
                rrects_j = rrects_j[keep]
                scores_j = scores_j[keep]

            boxlist_for_class = BoxList(bboxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("rrects", rrects_j)
            boxlist_for_class.add_field("scores", scores_j)

            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )
            result.append(boxlist_for_class)

        result = cat_boxlist(result)
        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        return result


def make_roi_box_post_processor(cfg):
    # use_fpn = cfg.MODEL.ROI_HEADS.USE_FPN

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    reg_angle_relative = cfg.MODEL.ROI_HEADS.BBOX_REG_ANGLE_RELATIVE
    box_coder = BoxCoder(weights=bbox_reg_weights, relative_angle=reg_angle_relative)

    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
    nms_thresh = cfg.MODEL.ROI_HEADS.NMS
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    use_nms = True
    if cfg.MODEL.MASKIOU_ON and "ROI_MASKIOU_HEAD" in cfg.MODEL and cfg.MODEL.ROI_MASKIOU_HEAD.USE_NMS:
        use_nms = False

    postprocessor = PostProcessor(
        box_coder,
        score_thresh,
        nms_thresh,
        detections_per_img,
        cls_agnostic_bbox_reg,
        use_nms
    )
    return postprocessor
