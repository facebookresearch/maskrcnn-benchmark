#pragma once
#include <torch/extension.h>

at::Tensor RROIAlign_forward_cuda(const at::Tensor& input,
                                const at::Tensor& rois,
                                const float spatial_scale,
                                const int pooled_height,
                                const int pooled_width,
                                const int sampling_ratio = 0);

at::Tensor RROIAlign_backward_cuda(const at::Tensor& grad,
                      const at::Tensor& rois,
                      const float spatial_scale,
                      const int pooled_height,
                      const int pooled_width,
                      const int batch_size,
                      const int channels,
                      const int height,
                      const int width,
                      const int sampling_ratio = 0);