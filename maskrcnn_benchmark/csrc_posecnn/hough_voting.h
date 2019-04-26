#pragma once

#include <vector>
#include <torch/extension.h>

#ifdef WITH_CUDA
#include "cuda/hough_voting_cuda.h"
#endif

// Interface for Python
std::vector<at::Tensor> hough_voting_forward(
    const at::Tensor& labels, const at::Tensor& masks, const at::Tensor& vertmap, const at::Tensor& extents, 
    const at::Tensor& meta_data, const at::Tensor& poses,
    const float inlierThreshold, const int skip_pixels
) 
{
  if (masks.type().is_cuda()) 
  {
#ifdef WITH_CUDA
    return hough_voting_forward_cuda(labels, masks, vertmap, extents, meta_data, poses,  
        inlierThreshold, skip_pixels);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
  // return hough_voting_forward_cpu(input1, input2);
}

// std::vector<at::Tensor> hough_voting_backward(const at::Tensor& grad) 
// {
//   if (grad.type().is_cuda()) {
// #ifdef WITH_CUDA
//     return hough_voting_backward_cuda(grad);
// #else
//     AT_ERROR("Not compiled with GPU support");
// #endif
//   }
//   AT_ERROR("Not implemented on the CPU");
//   // return hough_voting_backward_cpu(grad);
// }
