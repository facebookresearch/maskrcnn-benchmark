# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

# we need this for the custom ops to exist
import maskrcnn_benchmark._custom_ops   # noqa: F401

nms = torch.ops.maskrcnn_benchmark.nms

# nms.__doc__ = """
# This function performs Non-maximum suppresion"""
