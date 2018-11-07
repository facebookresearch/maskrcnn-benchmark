/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include "THCDeviceTensor.cuh"
//#include "THCDeviceTensorUtils.cuh"
#include "THCDeviceUtils.cuh"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)


template <typename Dtype>
__global__ void compute_flow_kernel(
    const int nthreads,
    THCDeviceTensor<Dtype, 2> boxes,
    THCDeviceTensor<Dtype, 4> output/*,
    const int height,
    const int width*/) {

  int N = boxes.getSize(0);
  int H = output.getSize(1);
  int W = output.getSize(2);
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int n = index % N;
    const int h = (index / N) % H;
    const int w = (index / (N * H)) % W;

    // int x0 = ScalarConvert<Dtype, int>::to(boxes[n][0]);
    // int y0 = ScalarConvert<Dtype, int>::to(boxes[n][1]);
    // int x1 = ScalarConvert<Dtype, int>::to(boxes[n][2]);
    // int y1 = ScalarConvert<Dtype, int>::to(boxes[n][3]);
    int x0 = int(boxes[n][0]);
    int y0 = int(boxes[n][1]);
    int x1 = int(boxes[n][2]);
    int y1 = int(boxes[n][3]);

    if ((w < x0) || (h < y0) || (w >= x1) || (h >= y1)) {
      output[n][h][w][0] = Dtype(-2);
      output[n][h][w][1] = Dtype(-2);
      continue;
    }

    int TO_REMOVE = 1;
    int box_width = x1 - x0 + TO_REMOVE;
    int box_height = y1 - y0 + TO_REMOVE;

    int xx0 = max(x0, 0);
    int yy0 = max(y0, 0);
    //int xx1 = std::min(x1, width);
    //int yy1 = std::min(y1, height);


    int lx = w - xx0;
    int ly = h - yy0;

    Dtype x = 2.f / (Dtype(box_width - 1)) * lx - 1.f;
    Dtype y = 2.f / (Dtype(box_height - 1)) * ly - 1.f;

    // get the corresponding input x, y co-ordinates from grid
    output[n][h][w][0] = x;
    output[n][h][w][1] = y;
  }
}

at::Tensor compute_flow_cuda(const at::Tensor& boxes,
                             const int height,
                             const int width) {
  AT_ASSERTM(boxes.type().is_cuda(), "boxes must be a CUDA tensor");

  auto num_rois = boxes.size(0);
  at::Tensor output = at::empty({num_rois, height, width, 2}, boxes.options());

  auto output_size = num_rois * height * width;
  dim3 grid(std::min(THCCeilDiv(output_size, 512L), 4096L));
  dim3 block(512);


  THCState *state = at::globalContext().lazyInitCUDA(); // TODO replace with getTHCState
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  //AT_DISPATCH_FLOATING_TYPES(boxes.type(), "compute_flow", [&] {
  /*
  using scalar_t = float;
  using THCTensor = THCudaTensor;
    THCDeviceTensor<scalar_t, 2> devBoxes = toDeviceTensor<scalar_t, 2>(state, (THCTensor *) boxes.unsafeGetTH(false));
    THCDeviceTensor<scalar_t, 4> devOutput = toDeviceTensor<scalar_t, 4>(state, (THCTensor *) output.unsafeGetTH(false));
    compute_flow_kernel<scalar_t><<<grid, block, 0, stream>>>(
      output_size,
      devBoxes,
      devOutput);

  //});
  */
  THCudaCheck(cudaGetLastError());
  return output;
}
