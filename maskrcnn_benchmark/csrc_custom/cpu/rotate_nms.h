#pragma once
#include <torch/extension.h>

at::Tensor rotate_nms_cpu(const at::Tensor& r_boxes,
                   const float nms_threshold, const int max_output = -1);

at::Tensor rotate_iou_matrix_cpu(const at::Tensor& r_boxes1,
				   const at::Tensor& r_boxes2);
