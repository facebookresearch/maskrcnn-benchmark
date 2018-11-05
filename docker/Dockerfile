ARG CUDA_VERSION=9.0
ARG CUDNN_VERSION=7

FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu16.04

# install basics
RUN apt-get update -y \
 && apt-get install -y git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ \
 && apt-get install -y --no-install-recommends libnccl2=2.3.5-2+cuda${CUDA_VERSION}

# Install Miniconda
RUN curl -so /miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p /miniconda \
 && rm /miniconda.sh

ENV PATH=/miniconda/bin:$PATH

# Create a Python 3.6 environment
RUN /miniconda/bin/conda install conda-build \
 && /miniconda/bin/conda create -y --name py36 python=3.6.7 \
 && /miniconda/bin/conda clean -ya

ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN conda install ipython
RUN pip install ninja yacs cython matplotlib

# Install PyTorch 1.0 Nightly
RUN conda install pytorch-nightly -c pytorch && conda clean -ya

# Install TorchVision master
RUN git clone https://github.com/pytorch/vision.git \
 && cd vision \
 && python setup.py install

# install pycocotools
RUN git clone https://github.com/cocodataset/cocoapi.git \
 && cd cocoapi/PythonAPI \
 && python setup.py build_ext install

# install PyTorch Detection
RUN git clone https://github.com/facebookresearch/maskrcnn-benchmark.git \
 && cd maskrcnn-benchmark \
 && python setup.py build develop

WORKDIR /maskrcnn-benchmark
