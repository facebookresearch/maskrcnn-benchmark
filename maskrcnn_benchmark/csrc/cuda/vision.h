// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once
#include <torch/extension.h>


at::Tensor SigmoidFocalLoss_forward_cuda(
		const at::Tensor& logits,
                const at::Tensor& targets,
		const int num_classes, 
		const float gamma, 
		const float alpha); 

at::Tensor SigmoidFocalLoss_backward_cuda(
			     const at::Tensor& logits,
                             const at::Tensor& targets,
			     const at::Tensor& d_losses,
			     const int num_classes,
			     const float gamma,
			     const float alpha);

at::Tensor ROIAlign_forward_cuda(const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const float spatial_scale,
                                 const int pooled_height,
                                 const int pooled_width,
                                 const int sampling_ratio);

at::Tensor ROIAlign_backward_cuda(const at::Tensor& grad,
                                  const at::Tensor& rois,
                                  const float spatial_scale,
                                  const int pooled_height,
                                  const int pooled_width,
                                  const int batch_size,
                                  const int channels,
                                  const int height,
                                  const int width,
                                  const int sampling_ratio);


std::tuple<at::Tensor, at::Tensor> ROIPool_forward_cuda(const at::Tensor& input,
                                const at::Tensor& rois,
                                const float spatial_scale,
                                const int pooled_height,
                                const int pooled_width);

at::Tensor ROIPool_backward_cuda(const at::Tensor& grad,
                                 const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const at::Tensor& argmax,
                                 const float spatial_scale,
                                 const int pooled_height,
                                 const int pooled_width,
                                 const int batch_size,
                                 const int channels,
                                 const int height,
                                 const int width);

at::Tensor nms_cuda(const at::Tensor boxes, float nms_overlap_thresh);


int deform_conv_forward_cuda(at::Tensor input, at::Tensor weight,
                             at::Tensor offset, at::Tensor output,
                             at::Tensor columns, at::Tensor ones, int kW,
                             int kH, int dW, int dH, int padW, int padH,
                             int dilationW, int dilationH, int group,
                             int deformable_group, int im2col_step);

int deform_conv_backward_input_cuda(at::Tensor input, at::Tensor offset,
                                    at::Tensor gradOutput, at::Tensor gradInput,
                                    at::Tensor gradOffset, at::Tensor weight,
                                    at::Tensor columns, int kW, int kH, int dW,
                                    int dH, int padW, int padH, int dilationW,
                                    int dilationH, int group,
                                    int deformable_group, int im2col_step);

int deform_conv_backward_parameters_cuda(
    at::Tensor input, at::Tensor offset, at::Tensor gradOutput,
    at::Tensor gradWeight,  // at::Tensor gradBias,
    at::Tensor columns, at::Tensor ones, int kW, int kH, int dW, int dH,
    int padW, int padH, int dilationW, int dilationH, int group,
    int deformable_group, float scale, int im2col_step);

void modulated_deform_conv_cuda_forward(
    at::Tensor input, at::Tensor weight, at::Tensor bias, at::Tensor ones,
    at::Tensor offset, at::Tensor mask, at::Tensor output, at::Tensor columns,
    int kernel_h, int kernel_w, const int stride_h, const int stride_w,
    const int pad_h, const int pad_w, const int dilation_h,
    const int dilation_w, const int group, const int deformable_group,
    const bool with_bias);

void modulated_deform_conv_cuda_backward(
    at::Tensor input, at::Tensor weight, at::Tensor bias, at::Tensor ones,
    at::Tensor offset, at::Tensor mask, at::Tensor columns,
    at::Tensor grad_input, at::Tensor grad_weight, at::Tensor grad_bias,
    at::Tensor grad_offset, at::Tensor grad_mask, at::Tensor grad_output,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
    int pad_w, int dilation_h, int dilation_w, int group, int deformable_group,
    const bool with_bias);

void deform_psroi_pooling_cuda_forward(
    at::Tensor input, at::Tensor bbox, at::Tensor trans, at::Tensor out,
    at::Tensor top_count, const int no_trans, const float spatial_scale,
    const int output_dim, const int group_size, const int pooled_size,
    const int part_size, const int sample_per_part, const float trans_std);

void deform_psroi_pooling_cuda_backward(
    at::Tensor out_grad, at::Tensor input, at::Tensor bbox, at::Tensor trans,
    at::Tensor top_count, at::Tensor input_grad, at::Tensor trans_grad,
    const int no_trans, const float spatial_scale, const int output_dim,
    const int group_size, const int pooled_size, const int part_size,
    const int sample_per_part, const float trans_std);


at::Tensor compute_flow_cuda(const at::Tensor& boxes,
                             const int height,
                             const int width);
