# MAX cm-lcm-layer

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator


class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.compatible_matrix = Parameter(torch.eye(num_classes-1, num_classes-1))
        self.local_compatible_matrix = Parameter(torch.eye(num_classes-1, num_classes-1))
        self.cm_weight = Parameter(torch.Tensor([0.5]))
        self.local_cm_weight = Parameter(torch.Tensor([0.5]))
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

    def compute_iou_as_similarity(self, a, b):
        # We need to add a trailing dimension so that max/min gives us a (N,N) matrix
        xA = torch.max(a[:,0].unsqueeze(1), b[:,0].unsqueeze(0))
        yA = torch.max(a[:,1].unsqueeze(1), b[:,1].unsqueeze(0))
        xB = torch.min(a[:,2].unsqueeze(1), b[:,2].unsqueeze(0))
        yB = torch.min(a[:,3].unsqueeze(1), b[:,3].unsqueeze(0))

        inter_w = xB - xA  + 1
        inter_w[inter_w < 0] = 0
        inter_h = yB - yA + 1
        inter_h[inter_h < 0] = 0

        intersection = inter_w * inter_h

        a_width = a[:,2]-a[:,0] + 1
        a_height = a[:,3]-a[:,1] + 1
        b_width = b[:,2]-b[:,0] + 1
        b_height = b[:,3]-b[:,1] + 1
        a_area = a_width.unsqueeze(1) * a_height.unsqueeze(1)
        b_area = b_width.unsqueeze(1) * b_height.unsqueeze(1)

        iou = intersection / (a_area + torch.t(b_area) - intersection)

        # set nan and +/- inf to 0
        iou[iou == float('inf')] = 0
        iou[iou != iou] = 0

        return iou

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)

        boxes_per_image = [len(box) for box in proposals]
        class_logits_split = class_logits.split(boxes_per_image, dim=0)
        q_values_splits = []
        for unaries, box_proposal in zip(class_logits_split,proposals):
            # add to split background and normal classes
            background_logits = unaries[:,0]
            unaries = unaries[:,1:]

            # compute global pairwise correlation
            q_values = unaries
            bbox_num, class_num = unaries.size()

            softmax_out = F.softmax(q_values, -1)

            # use sum as message passing
            # sum_softmax_out = torch.sum(softmax_out, dim = 0).repeat(bbox_num).view(bbox_num, class_num)
            # message_passing = F.softmax(sum_softmax_out-softmax_out)

            # use max as message passing
            message_passing = torch.max(softmax_out, dim = 0)[0].repeat(bbox_num).view(bbox_num, class_num)

            # Compatibility transform
            pairwise = torch.matmul(message_passing, self.compatible_matrix)
            # compatible_matrix_symmetric = torch.mm(self.compatible_matrix, self.compatible_matrix.t())
            # pairwise = torch.matmul(message_passing, compatible_matrix_symmetric)

            # compute local pairwise correlation
            # compute distance similarity using IOU
            boxes = box_proposal.bbox
            with torch.no_grad():
                dist_similarity = self.compute_iou_as_similarity(boxes,boxes)

            local_message_passing = torch.matmul(dist_similarity, softmax_out)
            # local_message_passing = F.softmax(local_message_passing, -1)
            # normalize local_message_passing
            # local_message_passing = local_message_passing/torch.t(torch.sum(local_message_passing, dim = 1)
            #                                                        .repeat(class_num).view(class_num, bbox_num))
            local_pairwise = torch.matmul(local_message_passing, self.local_compatible_matrix)

            # Adding unary potentials
            q_values = unaries + self.cm_weight * pairwise + self.local_cm_weight * local_pairwise  # * softmax_out
            # q_values_splits.append(q_values)
            q_values_splits.append(torch.cat((background_logits.unsqueeze(1), q_values),dim=1))

        class_logits = torch.cat(q_values_splits)

        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)
            # print(self.compatible_matrix)
            return x, result, {}

        loss_classifier, loss_box_reg = self.loss_evaluator(
            [class_logits], [box_regression]
        )
        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
        )


def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)
