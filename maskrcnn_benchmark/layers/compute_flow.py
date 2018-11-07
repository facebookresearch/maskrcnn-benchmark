# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from maskrcnn_benchmark import _C

# TODO make it work on the CPU
compute_flow = _C.compute_flow
