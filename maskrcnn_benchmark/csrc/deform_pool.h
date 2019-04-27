// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once
#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif


// Interface for Python
void deform_psroi_pooling_forward(
    at::Tensor input, 
    at::Tensor bbox, 
    at::Tensor trans, 
    at::Tensor out,
    at::Tensor top_count, 
    const int no_trans, 
    const float spatial_scale,
    const int output_dim, 
    const int group_size, 
    const int pooled_size,
    const int part_size, 
    const int sample_per_part, 
    const float trans_std)
{
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    return deform_psroi_pooling_cuda_forward(
        input, bbox, trans, out, top_count, 
        no_trans, spatial_scale, output_dim, group_size,
        pooled_size, part_size, sample_per_part, trans_std
    );
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}


void deform_psroi_pooling_backward(
    at::Tensor out_grad, 
    at::Tensor input, 
    at::Tensor bbox, 
    at::Tensor trans,
    at::Tensor top_count, 
    at::Tensor input_grad, 
    at::Tensor trans_grad,
    const int no_trans, 
    const float spatial_scale, 
    const int output_dim,
    const int group_size, 
    const int pooled_size, 
    const int part_size,
    const int sample_per_part, 
    const float trans_std) 
{
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    return deform_psroi_pooling_cuda_backward(
        out_grad, input, bbox, trans, top_count, input_grad, trans_grad,
        no_trans, spatial_scale, output_dim, group_size, pooled_size, 
        part_size, sample_per_part, trans_std
    );
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}
