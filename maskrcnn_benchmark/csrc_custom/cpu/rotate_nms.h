#pragma once
#include <torch/extension.h>

#include "nms_methods.h"

// at::Tensor rotate_nms_cpu(const at::Tensor& r_boxes,
//                    const float nms_threshold, const int max_output = -1);

at::Tensor rotate_nms_cpu(const at::Tensor& dets,
                   const at::Tensor& scores,
                   const float nms_threshold);

std::tuple<at::Tensor, at::Tensor> rotate_soft_nms_cpu(at::Tensor& dets,
                   at::Tensor& scores,
                   const float nms_thresh=0.3,
                   const float sigma=0.5,
                   const float score_thresh=0.001,
                   const int method=NMS_METHOD::GAUSSIAN
                   );

at::Tensor rotate_iou_matrix_cpu(const at::Tensor& r_boxes1,
				   const at::Tensor& r_boxes2);
