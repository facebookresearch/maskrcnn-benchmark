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
        self.compatible_matrix = Parameter(-torch.eye(num_classes, num_classes))
        # self.compatible_matrix = Parameter(-torch.eye(num_classes-1, num_classes-1))
        self.influence_weight = Parameter(torch.Tensor([1.0]))
        # stdv = 1. / math.sqrt(self.compatible_matrix.size(1))
        # self.compatible_matrix.data.uniform_(-stdv, stdv)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

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
        for unaries in class_logits_split:
            # add to split background and normal classes
            # background_logits = unaries[:,0]
            # unaries = unaries[:,1:]

            q_values = unaries
            bbox_num, class_num = unaries.size()

            # print('Unaries:', unaries)
            for i in range(10):
                softmax_out = F.softmax(q_values, -1)

                # use sum as message passing
                # sum_softmax_out = torch.sum(softmax_out, dim = 0).repeat(bbox_num).view(bbox_num, class_num)
                # message_passing = sum_softmax_out-softmax_out

                # use max as message passing
                message_passing = torch.max(softmax_out, dim = 0)[0].repeat(bbox_num).view(bbox_num, class_num)

                # use max as message passing, ignore background
                message_passing = self.influence_weight * message_passing

                # Compatibility transform
                pairwise = torch.matmul(message_passing, self.compatible_matrix)

                # Adding unary potentials
                q_values = unaries - pairwise
            q_values_splits.append(q_values)
            # q_values_splits.append(torch.cat((background_logits.unsqueeze(1), q_values),dim=1))

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
