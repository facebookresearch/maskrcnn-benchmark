// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include "rotate_rect_ops.h"

// TODO make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)


template <typename T>
__device__ inline void get_rotated_rect_bounding_box(const T* pts, int& leftMost, int& topMost,
  int& rightMost, int& bottomMost, const int width, const int height)
{
//  const T* P = pts;
//  leftMost = int(max(round(min(min(P[0],P[2]),min(P[4],P[6]))),0.0));
//  rightMost= int(min(round(max(max(P[0],P[2]),max(P[4],P[6]))),width-1.0));
//  topMost= int(max(round(min(min(P[1],P[3]),min(P[5],P[7]))),0.0));
//  bottomMost= int(min(round(max(max(P[1],P[3]),max(P[5],P[7]))),height-1.0));

  leftMost = int(max(min(min(pts[0], pts[2]), min(pts[4], pts[6])), 0.0));
  topMost = int(max(min(min(pts[1], pts[3]), min(pts[5], pts[7])), 0.0));
  rightMost = int(min(max(max(pts[0], pts[2]), max(pts[4], pts[6])) + 1, width - 1.0));
  bottomMost = int(min(max(max(pts[1], pts[3]), max(pts[5], pts[7])) + 1, height - 1.0));
}


template <typename T>
__global__ void RRoIPoolFForward(const int nthreads, const T* bottom_data,
    const float spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const T* bottom_rois, T* top_data, int* argmax_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * 6;
    int roi_batch_ind = offset_bottom_rois[0];

    T P[8];
    compute_roi_pool_pts(offset_bottom_rois, P, spatial_scale, pooled_height, pooled_width, ph, pw);

    int leftMost, topMost, rightMost, bottomMost;
    get_rotated_rect_bounding_box(P, leftMost, topMost, rightMost, bottomMost, width, height);

    T maxval = 0;
    int maxidx = -1;
    const T* offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;

    T AB[2];
    AB[0] = P[0] - P[2];
    AB[1] = P[1] - P[3];  
    T ABAB = AB[0]*AB[0] +AB[1]*AB[1];
    T AC[2];
    AC[0] = P[4] - P[2];
    AC[1] = P[5] - P[3];
    T ACAC = AC[0]*AC[0] + AC[1]*AC[1];

    for (int hh = topMost; hh < bottomMost+1; ++hh) {
      for (int ww = leftMost; ww < rightMost+1; ++ww) {
        T AP[2];
        AP[0] = ww - P[2];
        AP[1] = hh - P[3];
        T ABAP = AB[0]*AP[0] + AB[1]*AP[1];
        T ACAP = AC[0]*AP[0] + AC[1]*AP[1];
        if ( ABAP >= 1e-3 && (ABAB - ABAP) > -1e-3 && ACAP >= 1e-3 && (ACAC - ACAP) > -1e-3 )
        {
          int bottom_index = hh * width + ww;
          if (offset_bottom_data[bottom_index] > maxval) 
          {
            maxval = offset_bottom_data[bottom_index];
            maxidx = bottom_index;
          }
        }
      }
    }
    top_data[index] = maxval;
    argmax_data[index] = maxidx;

    // T bin_size_h = static_cast<T>(roi_height)
    //                    / static_cast<T>(pooled_height);
    // T bin_size_w = static_cast<T>(roi_width)
    //                    / static_cast<T>(pooled_width);

    // int hstart = static_cast<int>(floor(static_cast<T>(ph)
    //                                     * bin_size_h));
    // int wstart = static_cast<int>(floor(static_cast<T>(pw)
    //                                     * bin_size_w));
    // int hend = static_cast<int>(ceil(static_cast<T>(ph + 1)
    //                                  * bin_size_h));
    // int wend = static_cast<int>(ceil(static_cast<T>(pw + 1)
    //                                  * bin_size_w));

    // // Add roi offsets and clip to input boundaries
    // hstart = min(max(hstart + roi_start_h, 0), height);
    // hend = min(max(hend + roi_start_h, 0), height);
    // wstart = min(max(wstart + roi_start_w, 0), width);
    // wend = min(max(wend + roi_start_w, 0), width);
    // bool is_empty = (hend <= hstart) || (wend <= wstart);

    // // Define an empty pooling region to be zero
    // T maxval = is_empty ? 0 : -FLT_MAX;
    // // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    // int maxidx = -1;
    // const T* offset_bottom_data =
    //     bottom_data + (roi_batch_ind * channels + c) * height * width;
    // for (int h = hstart; h < hend; ++h) {
    //   for (int w = wstart; w < wend; ++w) {
    //     int bottom_index = h * width + w;
    //     if (offset_bottom_data[bottom_index] > maxval) {
    //       maxval = offset_bottom_data[bottom_index];
    //       maxidx = bottom_index;
    //     }
    //   }
    // }
    // top_data[index] = maxval;
    // argmax_data[index] = maxidx;
  }
}

template <typename T>
__global__ void RRoIPoolFBackward(const int nthreads, const T* top_diff,
    const int* argmax_data, const int num_rois, const float spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, T* bottom_diff,
    const T* bottom_rois) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * 6;
    int roi_batch_ind = offset_bottom_rois[0];
    int bottom_offset = (roi_batch_ind * channels + c) * height * width;
    int top_offset    = (n * channels + c) * pooled_height * pooled_width;
    const T* offset_top_diff = top_diff + top_offset;
    T* offset_bottom_diff = bottom_diff + bottom_offset;
    const int* offset_argmax_data = argmax_data + top_offset;

    int argmax = offset_argmax_data[ph * pooled_width + pw];
    if (argmax != -1) {
      atomicAdd(
          offset_bottom_diff + argmax,
          static_cast<T>(offset_top_diff[ph * pooled_width + pw]));

    }
  }
}

std::tuple<at::Tensor, at::Tensor> RROIPool_forward_cuda(const at::Tensor& input,
                                const at::Tensor& rois,
                                const float spatial_scale,
                                const int pooled_height,
                                const int pooled_width) {
  AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(rois.type().is_cuda(), "rois must be a CUDA tensor");

  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);

  auto output = at::empty({num_rois, channels, pooled_height, pooled_width}, input.options());
  auto output_size = num_rois * pooled_height * pooled_width * channels;
  auto argmax = at::zeros({num_rois, channels, pooled_height, pooled_width}, input.options().dtype(at::kInt));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(THCCeilDiv((long)output_size, 512L), 4096L));
  dim3 block(512);

  if (output.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return std::make_tuple(output, argmax);
  }

  AT_DISPATCH_FLOATING_TYPES(input.type(), "RROIPool_forward", [&] {
    RRoIPoolFForward<scalar_t><<<grid, block, 0, stream>>>(
         output_size,
         input.contiguous().data<scalar_t>(),
         spatial_scale,
         channels,
         height,
         width,
         pooled_height,
         pooled_width,
         rois.contiguous().data<scalar_t>(),
         output.data<scalar_t>(),
         argmax.data<int>());
  });
  THCudaCheck(cudaGetLastError());
  return std::make_tuple(output, argmax);
}

// TODO remove the dependency on input and use instead its sizes -> save memory
at::Tensor RROIPool_backward_cuda(const at::Tensor& grad,
                                 const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const at::Tensor& argmax,
                                 const float spatial_scale,
                                 const int pooled_height,
                                 const int pooled_width,
                                 const int batch_size,
                                 const int channels,
                                 const int height,
                                 const int width) {
  AT_ASSERTM(grad.type().is_cuda(), "grad must be a CUDA tensor");
  AT_ASSERTM(rois.type().is_cuda(), "rois must be a CUDA tensor");
  // TODO add more checks

  auto num_rois = rois.size(0);
  auto grad_input = at::zeros({batch_size, channels, height, width}, grad.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(THCCeilDiv((long)grad.numel(), 512L), 4096L));
  dim3 block(512);

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return grad_input;
  }

  AT_DISPATCH_FLOATING_TYPES(grad.type(), "RROIPool_backward", [&] {
    RRoIPoolFBackward<scalar_t><<<grid, block, 0, stream>>>(
         grad.numel(),
         grad.contiguous().data<scalar_t>(),
         argmax.data<int>(),
         num_rois,
         spatial_scale,
         channels,
         height,
         width,
         pooled_height,
         pooled_width,
         grad_input.data<scalar_t>(),
         rois.contiguous().data<scalar_t>());
  });
  THCudaCheck(cudaGetLastError());
  return grad_input;
}
