# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#!/usr/bin/env python

import glob
import os

import torch
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

requirements = ["torch", "torchvision"]

def get_extensions(extensions_dir, extension_name):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, extensions_dir)

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
            extension_name,
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


if __name__ == '__main__':
    from setuptools import find_packages
    from setuptools import setup

    extensions_dir = "maskrcnn_benchmark/csrc"
    extension_name = "maskrcnn_benchmark._C"
    ext_modules = []
    # ext_modules += get_extensions(extensions_dir, extension_name)

    custom_extensions_dir = "maskrcnn_benchmark/csrc_custom"
    custom_extension_name = "maskrcnn_benchmark._Custom"
    ext_modules += get_extensions(custom_extensions_dir, custom_extension_name)
    
    setup(
        name="maskrcnn_benchmark",
        version="0.1",
        # author="fmassa",
        # url="https://github.com/facebookresearch/maskrcnn-benchmark",
        description="object detection in pytorch",
        packages=find_packages(exclude=("configs", "tests",)),
        # install_requires=requirements,
        ext_modules=ext_modules,
        cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
    )
