# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
Variant of the resnet module that takes cfg as an argument.
Example usage. Strings may be specified in the config file.
    model = ResNet(
        "StemWithFixedBatchNorm",
        "BottleneckWithFixedBatchNorm",
        "ResNet50StagesTo4",
    )
Custom implementations may be written in user code and hooked in via the
`register_*` functions.
"""
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn

STEM_OUT_CHANNELS = 64
RES2_OUT_CHANNELS = 256

# ResNet stage specification
StageSpec = namedtuple(
    "StageSpec",
    [
        "index",  # Index of the stage, eg 1, 2, ..,. 5
        "block_count",  # Numer of residual blocks in the stage
        "return_features",  # True => return the last feature map from this stage
    ],
)

# -----------------------------------------------------------------------------
# Standard ResNet models
# -----------------------------------------------------------------------------
# ResNet-50 (including all stages)
ResNet50StagesTo5 = (
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((2, 3, False), (3, 4, False), (4, 6, False), (5, 3, True))
)
# ResNet-50 up to stage 4 (excludes stage 5)
ResNet50StagesTo4 = (
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((2, 3, False), (3, 4, False), (4, 6, True))
)
# ResNet-50-FPN (including all stages)
ResNet50FPNStagesTo5 = (
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((2, 3, True), (3, 4, True), (4, 6, True), (5, 3, True))
)


class ResNet(nn.Module):
    def __init__(self, cfg):
        super(ResNet, self).__init__()

        # If we want to use the cfg in forward(), then we should make a copy
        # of it and store it for later use:
        # self.cfg = cfg.clone()

        # Flag indicating that this module and its children can be
        # loaded from pretrained model state
        self.load_pretrained_state = True

        # Translate string names to implementations
        stem_module = _STEM_MODULES[cfg.MODEL.RESNETS.STEM_FUNC]
        stage_specs = _STAGE_SPECS[cfg.MODEL.BACKBONE.CONV_BODY]
        transformation_module = _TRANSFORMATION_MODULES[cfg.MODEL.RESNETS.TRANS_FUNC]

        # Construct the stem module
        self.stem = stem_module()

        # Constuct the specified ResNet stages
        num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        in_channels = STEM_OUT_CHANNELS
        stage2_bottleneck_channels = num_groups * width_per_group
        stage2_out_channels = RES2_OUT_CHANNELS
        self.stages = []
        self.return_features = {}
        for stage_spec in stage_specs:
            name = "res" + str(stage_spec.index)
            stage2_relative_factor = 2 ** (stage_spec.index - 2)
            bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
            out_channels = stage2_out_channels * stage2_relative_factor
            module = _make_stage(
                transformation_module,
                in_channels,
                bottleneck_channels,
                out_channels,
                stage_spec.block_count,
                num_groups,
                cfg.MODEL.RESNETS.STRIDE_IN_1X1,
                first_stride=int(stage_spec.index > 2) + 1,
            )
            in_channels = out_channels
            self.add_module(name, module)
            self.stages.append(name)
            self.return_features[name] = stage_spec.return_features

        # Optionally freeze (requires_grad=False) parts of the backbone
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)

    def _freeze_backbone(self, freeze_at):
        for stage_index in range(1, freeze_at + 1):
            if stage_index == 1:
                m = self.stem  # stage 1 is the stem
            else:
                m = getattr(self, "res" + str(stage_index))
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        for stage_name in self.stages:
            x = getattr(self, stage_name)(x)
            if self.return_features[stage_name]:
                outputs.append(x)
        return outputs


class ResNetHead(nn.Module):
    def __init__(
        self,
        block_module,
        stages,
        num_groups=1,
        width_per_group=64,
        stride_in_1x1=True,
        stride_init=None,
    ):
        in_channels = 1024  # TODO make it generic
        out_channels = 2048  # TODO make it generic
        bottleneck_channels = 512  # TODO make it generic
        super(ResNetHead, self).__init__()

        block_module = _TRANSFORMATION_MODULES[block_module]

        self.stages = []
        stride = stride_init
        for stage in stages:
            name = "res" + str(stage.index)
            if not stride:
                stride = int(stage.index > 2) + 1
            module = _make_stage(
                block_module,
                in_channels,
                bottleneck_channels,
                out_channels,
                stage.block_count,
                num_groups,
                stride_in_1x1,
                first_stride=stride,
            )
            stride = None
            self.add_module(name, module)
            self.stages.append(name)

    def forward(self, x):
        for stage in self.stages:
            x = getattr(self, stage)(x)
        return x


def _make_stage(
    transformation_module,
    in_channels,
    bottleneck_channels,
    out_channels,
    block_count,
    num_groups,
    stride_in_1x1,
    first_stride,
):
    blocks = []
    stride = first_stride
    for _ in range(block_count):
        blocks.append(
            transformation_module(
                in_channels,
                bottleneck_channels,
                out_channels,
                num_groups,
                stride_in_1x1,
                stride,
            )
        )
        stride = 1
        in_channels = out_channels
    return nn.Sequential(*blocks)


class FixedBatchNorm2d(nn.Module):
    """Equivalent to AffineChannel in (Caffe2) Detectron."""

    def __init__(self, n):
        super(FixedBatchNorm2d, self).__init__()
        self.register_buffer("scale", torch.ones(1, n, 1, 1))
        self.register_buffer("bias", torch.zeros(1, n, 1, 1))

    def forward(self, x):
        if x.dtype == torch.half:
            self.scale = self.scale.half()
            self.bias = self.bias.half()
        return x * self.scale + self.bias


class BottleneckWithFixedBatchNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups=1,
        stride_in_1x1=True,
        stride=1,
    ):
        super(BottleneckWithFixedBatchNorm, self).__init__()

        if in_channels != out_channels:
            self.branch1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )
            self.branch1_bn = FixedBatchNorm2d(out_channels)

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.branch2a = nn.Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
        )
        self.branch2a_bn = FixedBatchNorm2d(bottleneck_channels)
        # TODO: specify init for the above

        self.branch2b = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1,
            bias=False,
            groups=num_groups,
        )
        self.branch2b_bn = FixedBatchNorm2d(bottleneck_channels)

        self.branch2c = nn.Conv2d(
            bottleneck_channels, out_channels, kernel_size=1, bias=False
        )
        self.branch2c_bn = FixedBatchNorm2d(out_channels)

    def forward(self, x):
        residual = x

        out = self.branch2a(x)
        out = self.branch2a_bn(out)
        out = F.relu_(out)

        out = self.branch2b(out)
        out = self.branch2b_bn(out)
        out = F.relu_(out)

        out0 = self.branch2c(out)
        out = self.branch2c_bn(out0)

        if hasattr(self, "branch1"):
            residual = self.branch1(x)
            residual = self.branch1_bn(residual)

        out += residual
        out = F.relu_(out)

        return out


class StemWithFixedBatchNorm(nn.Module):
    def __init__(self):
        super(StemWithFixedBatchNorm, self).__init__()

        self.conv1 = nn.Conv2d(
            3, STEM_OUT_CHANNELS, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.conv1_bn = FixedBatchNorm2d(STEM_OUT_CHANNELS)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x


_TRANSFORMATION_MODULES = {"BottleneckWithFixedBatchNorm": BottleneckWithFixedBatchNorm}

_STEM_MODULES = {"StemWithFixedBatchNorm": StemWithFixedBatchNorm}

_STAGE_SPECS = {
    "R-50-C4": ResNet50StagesTo4,
    "R-50-C5": ResNet50StagesTo5,
    "R-50-FPN": ResNet50FPNStagesTo5,
}


def register_transformation_module(module_name, module):
    _register_generic(_TRANSFORMATION_MODULES, module_name, module)


def register_stem_module(module_name, module):
    _register_generic(_STEM_MODULES, module_name, module)


def register_stage_spec(stage_spec_name, stage_spec):
    _register_generic(_STAGE_SPECS, stage_spec_name, stage_spec)


def _register_generic(module_dict, module_name, module):
    assert module_name not in module_dict
    module_dict[module_name] = module
