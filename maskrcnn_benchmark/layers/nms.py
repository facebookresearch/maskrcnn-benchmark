# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from ._utils import _C
from torch.utils.cpp_extension import load
import os
import glob

ext_dir = os.path.join(os.path.abspath("../csrc"))
main_file = glob.glob(os.path.join(ext_dir, "*.cpp"))
source_cpu = glob.glob(os.path.join(ext_dir, "cpu", "*.cpp"))
source_cuda = glob.glob(os.path.join(ext_dir, "cuda", "*.cu"))
sources = main_file + source_cpu  + source_cuda
cuda_flags = [
    "-DCUDA_HAS_FP16=1",
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
]

C_functions = load("vision", sources, extra_cuda_cflags=cuda_flags, extra_include_paths=[ext_dir], with_cuda=True)


nms = C_functions.nms
# nms.__doc__ = """
# This function performs Non-maximum suppresion"""
