#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import glob
import os
import copy

import torch
from setuptools import find_packages
from setuptools import setup
import distutils.command.build

from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

requirements = ["torch", "torchvision"]


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "maskrcnn_benchmark", "csrc")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension

    custom_ops_sources = [os.path.join(extensions_dir, "custom_ops", "custom_ops.cpp"),
                          os.path.join(extensions_dir, "cpu", "nms_cpu.cpp"),
                          os.path.join(extensions_dir, "cpu", "ROIAlign_cpu.cpp")]
    custom_ops_sources_cuda = [os.path.join(extensions_dir, "cuda", "nms.cu"),
                               os.path.join(extensions_dir, "cuda", "ROIAlign_cuda.cu")]
    custom_ops_libraries = ["opencv_core", "opencv_imgproc", "opencv_imgcodecs"]
    extra_compile_args = {"cxx": []}
    define_macros = []

    if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv("FORCE_CUDA", "0") == "1":
        extension = CUDAExtension
        sources += source_cuda
        custom_ops_sources += custom_ops_sources_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "maskrcnn_benchmark._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        ),
        extension(
            "maskrcnn_benchmark.lib.custom_ops",
            sources=custom_ops_sources,
            include_dirs=copy.deepcopy(include_dirs),
            define_macros=copy.deepcopy(define_macros),
            extra_compile_args=copy.deepcopy(extra_compile_args),
            libraries=custom_ops_libraries
        ),
    ]

    return ext_modules


class rename_custom_ops_lib(distutils.command.build.build):
    def run(self):
        inst_dir = os.path.join(self.build_lib, 'maskrcnn_benchmark', 'lib')
        lib_suffix = os.path.basename(torch._C.__file__).split('.', 1)[1]  # there must be a better way
        ext = lib_suffix.rsplit('.', 1)[1]
        os.rename(os.path.join(inst_dir, 'custom_ops.' + lib_suffix),
                  os.path.join(inst_dir, 'libmaskrcnn_benchmark_customops.' + ext))


class build(distutils.command.build.build):
    sub_commands = distutils.command.build.build.sub_commands + [
        ('rename_custom_ops_lib', lambda self: True),
    ]

setup(
    name="maskrcnn_benchmark",
    version="0.1",
    author="fmassa",
    url="https://github.com/facebookresearch/maskrcnn-benchmark",
    description="object detection in pytorch",
    packages=find_packages(exclude=("configs", "tests",)),
    # install_requires=requirements,
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension,
              "rename_custom_ops_lib": rename_custom_ops_lib,
              "build": build},
)
