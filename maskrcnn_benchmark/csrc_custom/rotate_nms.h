#pragma once

#include <vector>
#include <torch/extension.h>

#include "cpu/rotate_nms.h"

#ifdef WITH_CUDA
#include "cuda/rotate_nms_cuda.h"
#endif

// Interface for Python
at::Tensor rotate_nms(
    const at::Tensor& r_boxes, const float nms_threshold, const int max_output
)
{
  if (r_boxes.type().is_cuda())
  {
#ifdef WITH_CUDA
    return rotate_nms_cuda(r_boxes, nms_threshold, max_output);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }

  return rotate_nms_cpu(r_boxes, nms_threshold, max_output);
}

// Interface for Python
at::Tensor rotate_iou_matrix(
    const at::Tensor& r_boxes1, const at::Tensor& r_boxes2
)
{
  if (r_boxes1.type().is_cuda())
  {
#ifdef WITH_CUDA
    return rotate_iou_matrix_cuda(r_boxes1, r_boxes2);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }

  return rotate_iou_matrix_cpu(r_boxes1, r_boxes2);
}
