import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from .loss import make_rpn_loss_evaluator
from .anchor_generator import make_anchor_generator
from .inference import make_rpn_postprocessor


class RetinaNetHead(torch.nn.Module):
    """
    Adds a RetinNet head with classification and regression heads
    """

    def __init__(self, cfg):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RetinaNetHead, self).__init__()
        anchor_generator = make_anchor_generator(cfg)
        # TODO: Add the sigmoid version first.
        num_classes = cfg.RETINANET.NUM_CLASSES - 1
        in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        num_anchors = anchor_generator.num_anchors_per_location()[0]

        self.cls_tower = []
        self.bbox_tower = []
        cls_tower = []
        bbox_tower = []
        for i in cfg.RETINANET.NUM_CONVS:
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )

        self.add_module(cls_tower, nn.Sequential(*cls_tower))
        self.add_module(bbox_tower, nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=1, stride=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels,  num_anchors * 4, kernel_size=1, stride=1
        )

        # Initialization
        for m in [self.cls_tower, self.bbox_tower, self.cls_logits,
                  self.bbox_pred]:
            for l in m.module():
                torch.nn.init.normal_(l.weight, std=0.01)
                torch.nn.init.constant_(l.bias, 0)

        # retinanet_bias_init
        prior_prob = cfg.RETINANET.PRIOR_PROB
        bias_value = -np.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            logits.append(self.cls_logits(self.cls_tower(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_twoer(feature)))
        return logits, bbox_reg


class RetinaNetModule(torch.nn.Module):
    """
    Module for RetinaNet computation. Takes feature maps from the backbone and RPN
    proposals and losses.
    """

    def __init__(self, cfg):
        super(RetinaNetModule, self).__init__()

        self.cfg = cfg.clone()

        anchor_generator = make_anchor_generator(cfg)
        head = RetinaNetHead(cfg)
        '''
        box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        box_selector_train = make_rpn_postprocessor(cfg, box_coder, is_train=True)
        box_selector_test = make_rpn_postprocessor(cfg, box_coder, is_train=False)

        loss_evaluator = make_rpn_loss_evaluator(cfg, box_coder)

        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_train = box_selector_train
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        '''
    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls, box_regression = self.head(features)
        anchors = self.anchor_generator(images, features)

        if self.training:
            return self._forward_train(anchors, box_cls, box_regression, targets)
        else:
            return self._forward_test(anchors, box_cls, box_regression)

    def _forward_train(self, anchors, objectness, rpn_box_regression, targets):
        if self.cfg.MODEL.RPN_ONLY:
            # When training an RPN-only model, the loss is determined by the
            # predicted objectness and rpn_box_regression values and there is
            # no need to transform the anchors into predicted boxes; this is an
            # optimization that avoids the unnecessary transformation.
            boxes = anchors
        else:
            # For end-to-end models, anchors must be transformed into boxes and
            # sampled into a training batch.
            with torch.no_grad():
                boxes = self.box_selector_train(
                    anchors, objectness, rpn_box_regression, targets
                )
        loss_objectness, loss_rpn_box_reg = self.loss_evaluator(
            anchors, objectness, rpn_box_regression, targets
        )
        losses = {
            "loss_retina_cls": loss_objectness,
            "loss_retina_reg": loss_rpn_box_reg,
        }
        return boxes, losses

    def _forward_test(self, anchors, objectness, rpn_box_regression):
        boxes = self.box_selector_test(anchors, objectness, rpn_box_regression)
        if self.cfg.MODEL.RPN_ONLY:
            # For end-to-end models, the RPN proposals are an intermediate state
            # and don't bother to sort them in decreasing score order. For RPN-only
            # models, the proposals are the final output and we return them in
            # high-to-low confidence order.
            inds = [
                box.get_field("objectness").sort(descending=True)[1] for box in boxes
            ]
            boxes = [box[ind] for box, ind in zip(boxes, inds)]
        return boxes, {}


def build_retinanet(cfg):
    return RetinaNetModule(cfg)
