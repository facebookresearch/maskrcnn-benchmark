#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import glob
import os
import sys
import errno

import torch
from setuptools import find_packages
from setuptools import setup
import distutils.command.build
import subprocess
import shutil

from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

requirements = ["torch", "torchvision"]


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "maskrcnn_benchmark", "csrc")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
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
        )
    ]

    return ext_modules


class build_custom_ops(distutils.command.build.build):
    def run(self):
        build_dir = os.path.join('build', 'custom_ops')
        inst_dir = os.path.join(self.build_lib, 'maskrcnn_benchmark', 'lib')
        mkdir(build_dir)
        torch_path = os.path.dirname(torch.__file__)
        env = {'Torch_DIR': os.path.join(torch_path , 'share', 'cmake', 'Torch'),
               'Caffe2_DIR': os.path.join(torch_path , 'share', 'cmake', 'Caffe2')}
        os.environ.update(env)
        if subprocess.call(['cmake', '../../maskrcnn_benchmark/csrc/custom_ops'], cwd=build_dir) != 0:
            print("failed to build custom ops")
            sys.exit(1)
        if subprocess.call(['make'], cwd=build_dir) != 0:
            print("failed to build custom ops")
            sys.exit(1)
        mkdir(inst_dir)
        ext = 'so'  # different for OS X and Windows
        shutil.copy(os.path.join(build_dir, 'libmaskrcnn_benchmark_customops.' + ext), inst_dir)


class build(distutils.command.build.build):
    sub_commands = distutils.command.build.build.sub_commands + [
        ('build_custom_ops', lambda self: True),
    ]


setup(
    name="maskrcnn_benchmark",
    version="0.1",
    author="fmassa",
    url="https://github.com/facebookresearch/maskrnn-benchmark",
    description="object detection in pytorch",
    packages=find_packages(exclude=("configs", "tests",)),
    # install_requires=requirements,
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension,
              "build_custom_ops": build_custom_ops,
              "build": build},
)
