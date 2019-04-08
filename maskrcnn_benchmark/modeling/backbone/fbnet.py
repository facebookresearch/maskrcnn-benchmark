from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import json
import logging
from collections import OrderedDict

from . import (
    fbnet_builder as mbuilder,
    fbnet_modeldef as modeldef,
)
import torch.nn as nn
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.rpn import rpn
from maskrcnn_benchmark.modeling import poolers


logger = logging.getLogger(__name__)


def create_builder(cfg):
    bn_type = cfg.MODEL.FBNET.BN_TYPE
    if bn_type == "gn":
        bn_type = (bn_type, cfg.GROUP_NORM.NUM_GROUPS)
    factor = cfg.MODEL.FBNET.SCALE_FACTOR

    arch = cfg.MODEL.FBNET.ARCH
    arch_def = cfg.MODEL.FBNET.ARCH_DEF
    if len(arch_def) > 0:
        arch_def = json.loads(arch_def)
    if arch in modeldef.MODEL_ARCH:
        if len(arch_def) > 0:
            assert (
                arch_def == modeldef.MODEL_ARCH[arch]
            ), "Two architectures with the same name {},\n{},\n{}".format(
                arch, arch_def, modeldef.MODEL_ARCH[arch]
            )
        arch_def = modeldef.MODEL_ARCH[arch]
    else:
        assert arch_def is not None and len(arch_def) > 0
    arch_def = mbuilder.unify_arch_def(arch_def)

    rpn_stride = arch_def.get("rpn_stride", None)
    if rpn_stride is not None:
        assert (
            cfg.MODEL.RPN.ANCHOR_STRIDE[0] == rpn_stride
        ), "Needs to set cfg.MODEL.RPN.ANCHOR_STRIDE to {}, got {}".format(
            rpn_stride, cfg.MODEL.RPN.ANCHOR_STRIDE
        )
    width_divisor = cfg.MODEL.FBNET.WIDTH_DIVISOR
    dw_skip_bn = cfg.MODEL.FBNET.DW_CONV_SKIP_BN
    dw_skip_relu = cfg.MODEL.FBNET.DW_CONV_SKIP_RELU

    logger.info(
        "Building fbnet model with arch {} (without scaling):\n{}".format(
            arch, arch_def
        )
    )

    builder = mbuilder.FBNetBuilder(
        width_ratio=factor,
        bn_type=bn_type,
        width_divisor=width_divisor,
        dw_skip_bn=dw_skip_bn,
        dw_skip_relu=dw_skip_relu,
    )

    return builder, arch_def


def _get_trunk_cfg(arch_def):
    """ Get all stages except the last one """
    num_stages = mbuilder.get_num_stages(arch_def)
    trunk_stages = arch_def.get("backbone", range(num_stages - 1))
    ret = mbuilder.get_blocks(arch_def, stage_indices=trunk_stages)
    return ret


class FBNetTrunk(nn.Module):
    def __init__(
        self, builder, arch_def, dim_in,
    ):
        super(FBNetTrunk, self).__init__()
        self.first = builder.add_first(arch_def["first"], dim_in=dim_in)
        trunk_cfg = _get_trunk_cfg(arch_def)
        self.stages = builder.add_blocks(trunk_cfg["stages"])

    # return features for each stage
    def forward(self, x):
        y = self.first(x)
        y = self.stages(y)
        ret = [y]
        return ret


@registry.BACKBONES.register("FBNet")
def add_conv_body(cfg, dim_in=3):
    builder, arch_def = create_builder(cfg)

    body = FBNetTrunk(builder, arch_def, dim_in)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = builder.last_depth

    return model


def _get_rpn_stage(arch_def, num_blocks):
    rpn_stage = arch_def.get("rpn")
    ret = mbuilder.get_blocks(arch_def, stage_indices=rpn_stage)
    if num_blocks > 0:
        logger.warn('Use last {} blocks in {} as rpn'.format(num_blocks, ret))
        block_count = len(ret["stages"])
        assert num_blocks <= block_count, "use block {}, block count {}".format(
            num_blocks, block_count
        )
        blocks = range(block_count - num_blocks, block_count)
        ret = mbuilder.get_blocks(ret, block_indices=blocks)
    return ret["stages"]


class FBNetRPNHead(nn.Module):
    def __init__(
        self, cfg, in_channels, builder, arch_def,
    ):
        super(FBNetRPNHead, self).__init__()
        assert in_channels == builder.last_depth

        rpn_bn_type = cfg.MODEL.FBNET.RPN_BN_TYPE
        if len(rpn_bn_type) > 0:
            builder.bn_type = rpn_bn_type

        use_blocks = cfg.MODEL.FBNET.RPN_HEAD_BLOCKS
        stages = _get_rpn_stage(arch_def, use_blocks)

        self.head = builder.add_blocks(stages)
        self.out_channels = builder.last_depth

    def forward(self, x):
        x = [self.head(y) for y in x]
        return x


@registry.RPN_HEADS.register("FBNet.rpn_head")
def add_rpn_head(cfg, in_channels, num_anchors):
    builder, model_arch = create_builder(cfg)
    builder.last_depth = in_channels

    assert in_channels == builder.last_depth
    # builder.name_prefix = "[rpn]"

    rpn_feature = FBNetRPNHead(cfg, in_channels, builder, model_arch)
    rpn_regressor = rpn.RPNHeadConvRegressor(
        cfg, rpn_feature.out_channels, num_anchors)
    return nn.Sequential(rpn_feature, rpn_regressor)


def _get_head_stage(arch, head_name, blocks):
    # use default name 'head' if the specific name 'head_name' does not existed
    if head_name not in arch:
        head_name = "head"
    head_stage = arch.get(head_name)
    ret = mbuilder.get_blocks(arch, stage_indices=head_stage, block_indices=blocks)
    return ret["stages"]


# name mapping for head names in arch def and cfg
ARCH_CFG_NAME_MAPPING = {
    "bbox": "ROI_BOX_HEAD",
    "kpts": "ROI_KEYPOINT_HEAD",
    "mask": "ROI_MASK_HEAD",
}


class FBNetROIHead(nn.Module):
    def __init__(
        self, cfg, in_channels, builder, arch_def,
        head_name, use_blocks, stride_init, last_layer_scale,
    ):
        super(FBNetROIHead, self).__init__()
        assert in_channels == builder.last_depth
        assert isinstance(use_blocks, list)

        head_cfg_name = ARCH_CFG_NAME_MAPPING[head_name]
        self.pooler = poolers.make_pooler(cfg, head_cfg_name)

        stage = _get_head_stage(arch_def, head_name, use_blocks)

        assert stride_init in [0, 1, 2]
        if stride_init != 0:
            stage[0]["block"][3] = stride_init
        blocks = builder.add_blocks(stage)

        last_info = copy.deepcopy(arch_def["last"])
        last_info[1] = last_layer_scale
        last = builder.add_last(last_info)

        self.head = nn.Sequential(OrderedDict([
            ("blocks", blocks),
            ("last", last)
        ]))

        self.out_channels = builder.last_depth

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.head(x)
        return x


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FBNet.roi_head")
def add_roi_head(cfg, in_channels):
    builder, model_arch = create_builder(cfg)
    builder.last_depth = in_channels
    # builder.name_prefix = "_[bbox]_"

    return FBNetROIHead(
        cfg, in_channels, builder, model_arch,
        head_name="bbox",
        use_blocks=cfg.MODEL.FBNET.DET_HEAD_BLOCKS,
        stride_init=cfg.MODEL.FBNET.DET_HEAD_STRIDE,
        last_layer_scale=cfg.MODEL.FBNET.DET_HEAD_LAST_SCALE,
    )


@registry.ROI_KEYPOINT_FEATURE_EXTRACTORS.register("FBNet.roi_head_keypoints")
def add_roi_head_keypoints(cfg, in_channels):
    builder, model_arch = create_builder(cfg)
    builder.last_depth = in_channels
    # builder.name_prefix = "_[kpts]_"

    return FBNetROIHead(
        cfg, in_channels, builder, model_arch,
        head_name="kpts",
        use_blocks=cfg.MODEL.FBNET.KPTS_HEAD_BLOCKS,
        stride_init=cfg.MODEL.FBNET.KPTS_HEAD_STRIDE,
        last_layer_scale=cfg.MODEL.FBNET.KPTS_HEAD_LAST_SCALE,
    )


@registry.ROI_MASK_FEATURE_EXTRACTORS.register("FBNet.roi_head_mask")
def add_roi_head_mask(cfg, in_channels):
    builder, model_arch = create_builder(cfg)
    builder.last_depth = in_channels
    # builder.name_prefix = "_[mask]_"

    return FBNetROIHead(
        cfg, in_channels, builder, model_arch,
        head_name="mask",
        use_blocks=cfg.MODEL.FBNET.MASK_HEAD_BLOCKS,
        stride_init=cfg.MODEL.FBNET.MASK_HEAD_STRIDE,
        last_layer_scale=cfg.MODEL.FBNET.MASK_HEAD_LAST_SCALE,
    )
