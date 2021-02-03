
# maskrcnn-benchmark Dockerfile
#
# Updated and it worked in Feb 2021

# If you've never worked with docker before:
#
# This script installs a lot of software in a docker image. This image can be used to
# create a docker container in which you can execute code.

# I also included ROS and OpenCV installs in here. This will increase the size of the
# image to >10GB. Feel free to delete this if you don't need it.

# You can also choose whether to install the upstream maskrcnn-benchmark or my fork where
# I tried to fix some issues according to what people discussed in the issues.


# === Useful commands: ===

# I'm using nvidia-docker for all of this since I want to use my GPU inside the container.

# Build an image from the Dockerfile:
# $ cd < cloned maskrcnn-benchmark repository >
# $ nvidia-docker build -t maskrcnn-benchmark-ros docker/
#                          < name for the image >

# Start new container from image:
# $ nvidia-docker run -it --entrypoint bash maskrcnn-benchmark

# Enter running container:
# $ nvidia-docker exec -it maskrcnn-benchmark bash


ARG CUDA="10.0"
ARG CUDNN="7"
# cuda 9 + cudnn 7 had problems with cudnn on NVIDIA Turing Hardware (RTX Gen 1)
# cuda 10 + cudnn 7 works for Turing

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu16.04

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# install basics
RUN apt-get update -y \
 && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ \
 && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev

# Install Miniconda
RUN curl -L -so /miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p /miniconda \
 && rm /miniconda.sh

ENV PATH=/miniconda/bin:$PATH

# Create a Python 3.6 environment
RUN /miniconda/bin/conda install -y conda-build \
 && /miniconda/bin/conda create -y --name py36 python=3.6.7 \
 && /miniconda/bin/conda clean -ya

ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN conda install -y ipython
RUN pip install requests ninja yacs cython matplotlib opencv-python tqdm
RUN pip install attrs scikit-image
RUN pip3 install cityscapesScripts

# Install PyTorch 1.0 Nightly
ARG CUDA
RUN conda install pytorch-nightly=1.0 cudatoolkit=${CUDA} -c pytorch \
 && conda clean -ya

# Install TorchVision master
RUN git clone https://github.com/pytorch/vision.git \
 && cd vision \
 && git checkout "v0.2.2_branch" \
 && python setup.py install

# install pycocotools
RUN git clone https://github.com/cocodataset/cocoapi.git \
 && cd cocoapi/PythonAPI \
 && python setup.py build_ext install

# install apex
RUN git clone https://github.com/NVIDIA/apex.git \
 && cd apex \
 && git checkout f3a960f80244cf9e80558ab30f7f7e8cbf03c0a0 \
 && python setup.py install --cuda_ext --cpp_ext


# install Detection
# TO USE THE UPSTREAM REPO WHICH DIDN'T WORK FOR ME:
# The three lines with sed try to fix at least some problems. Look at README.md for more info.
# ARG FORCE_CUDA="1"
# ENV FORCE_CUDA=${FORCE_CUDA}
# RUN git clone https://github.com/facebookresearch/maskrcnn-benchmark.git \
#  && cd maskrcnn-benchmark \
#  && sed -i"" "s/torch\.bool/torch.uint8/g" maskrcnn_benchmark/structures/segmentation_mask.py \
#  && sed -i"" "s/torch\.bool/torch.uint8/g" maskrcnn_benchmark/modeling/rpn/inference.py \
#  && sed -i"" "s/torch\.bool/torch.uint8/g" maskrcnn_benchmark/modeling/balanced_positive_negative_sampler.py \
#  && python setup.py build develop

# USE FIXED REPO:
ARG FORCE_CUDA="1"
ENV FORCE_CUDA=${FORCE_CUDA}
RUN git clone https://github.com/montioo/maskrcnn-benchmark.git \
 && cd maskrcnn-benchmark \
 && python setup.py build develop



# === ROS ===
# only necessary if you want to use maskrcnn-benchmark with ros

RUN apt-get install -y apt-transport-https

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu xenial main" > /etc/apt/sources.list.d/ros-latest.list'

RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    python-rosdep \
    python-rosinstall \
    python-vcstools \
    && rm -rf /var/lib/apt/lists/*

# bootstrap rosdep
RUN rosdep init && \
  rosdep update --rosdistro kinetic

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-kinetic-ros-base \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
 && apt-get install -y vim python-rospy python-opencv

RUN apt-get install -y python3-pip python3-yaml
RUN pip3 install rospkg catkin_pkg
RUN apt-get install -y python-catkin-tools python3-dev python3-numpy
RUN apt-get install -y ros-kinetic-cv-bridge
RUN apt-get install -y ros-kinetic-vision-opencv



# === Open CV ===

RUN apt-get install -y gcc g++
RUN apt-get install -y python3-dev python3-numpy
RUN apt-get install -y libavcodec-dev libavformat-dev libswscale-dev
RUN apt-get install -y libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev
RUN apt-get install -y libgtk-3-dev

RUN git clone https://github.com/opencv/opencv.git \
 && cd opencv \
 && git checkout 3.4 \
 && mkdir build \
 && cd build \
 && cmake ../ -DPYTHON_EXECUTABLE=/miniconda/envs/py36/bin/python3 -DPYTHON_INCLUDE_DIR=/miniconda/envs/py36/include/python3.6m -DPYTHON_LIBRARY=/miniconda/envs/py36/lib/libpython3.6m.so \
 && make -j 16

RUN conda install -y boost

RUN echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc
RUN echo "set -o vi" >> ~/.bashrc

WORKDIR /maskrcnn-benchmark
