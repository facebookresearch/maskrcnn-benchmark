#pragma once
#include <torch/extension.h>

at::Tensor rotate_nms_cuda(
    const at::Tensor& r_boxes, const float nms_threshold, const int max_output
);
at::Tensor rotate_iou_matrix_cuda(
    const at::Tensor& r_boxes1, const at::Tensor& r_boxes2
);
