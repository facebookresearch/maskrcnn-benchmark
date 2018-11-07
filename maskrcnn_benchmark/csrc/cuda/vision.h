/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 */

#pragma once
#include <torch/extension.h>


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


at::Tensor compute_flow_cuda(const at::Tensor& boxes,
                             const int height,
                             const int width);

at::Tensor match_proposals_cuda(at::Tensor match_quality_matrix,bool include_low_quality_matches, float low_th, float high_th);

at::Tensor generate_mask_targets_cuda( at::Tensor dense_vector, std::vector<std::vector<at::Tensor>> polygons, at::Tensor anchors, int mask_size);

std::vector<at::Tensor> box_encode_cuda( at::Tensor boxes, at::Tensor anchors, float wx, float wy, float ww, float wh);

at::Tensor box_iou_cuda( at::Tensor box1, at::Tensor box2);

