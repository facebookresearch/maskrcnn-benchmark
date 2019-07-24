# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from ._utils import _C
from maskrcnn_benchmark import _C

try:
  from apex import amp
  # Only valid with fp32 inputs - give AMP the hint
  nms = amp.float_function(_C.nms)
except Exception as e:
  print("Couldn't load apex, because you are running on cpu probably, and couldn't detect cuda !")
  nms = _C.nms

# nms.__doc__ = """
# This function performs Non-maximum suppresion"""
