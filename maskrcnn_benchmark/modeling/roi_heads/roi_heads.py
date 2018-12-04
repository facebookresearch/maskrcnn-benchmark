# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .box_head.box_head import build_roi_box_head
from .mask_head.mask_head import build_roi_mask_head

class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor
        if cfg.MODEL.VERTEX_ON and cfg.MODEL.ROI_VERTEX_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.vertex.feature_extractor = self.box.feature_extractor
        if cfg.MODEL.POSE_ON and cfg.MODEL.ROI_POSE_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.pose.feature_extractor = self.box.feature_extractor

    def forward(self, features, proposals, targets=None):
        losses = {}
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        x, bbox_pred, detections, loss_box = self.box(features, proposals, targets)
        losses.update(loss_box)

        # detections_list = [detections]
        if self.cfg.MODEL.MASK_ON:
            mask_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                mask_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, mask_logits, detections, loss_mask = self.mask(mask_features, detections, targets)
            losses.update(loss_mask)

            # if not self.training:
            #     detections = mask_detections

        if self.cfg.MODEL.VERTEX_ON:
            vertex_features = features
            if (
                self.training
                and self.cfg.MODEL.ROI_VERTEX_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                vertex_features = x
            # use
            x, vertex_pred, detections, loss_vertex = self.vertex(vertex_features, detections, targets)
            losses.update(loss_vertex)

            # if not self.training:
            #     detections = vertex_detections

        if self.cfg.MODEL.POSE_ON:
            # resize mask and vertexes to original resolution (resolution size can be found in
            # detections/proposals var). mask out all values of the vertexes where mask value < 0.5
            # possibly filter out masks where score < 0.9
            # these will be our inputs to hough vote layer
            pose_features = features

            if (
                self.training
                and self.cfg.MODEL.ROI_POSE_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                pose_features = x
            # use
            x, pose_pred, detections, loss_pose = self.pose(pose_features, detections, targets)
            losses.update(loss_pose)

        return x, detections, losses


def build_roi_heads(cfg):
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(("box", build_roi_box_head(cfg)))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(("mask", build_roi_mask_head(cfg)))
    if cfg.MODEL.VERTEX_ON:
        from .vertex_head.vertex_head import build_roi_vertex_head
        roi_heads.append(("vertex", build_roi_vertex_head(cfg)))
    if cfg.MODEL.POSE_ON:
        from .pose_head.pose_head import build_roi_pose_head
        roi_heads.append(("pose", build_roi_pose_head(cfg)))

    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads
