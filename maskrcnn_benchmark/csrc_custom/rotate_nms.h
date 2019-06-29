#pragma once

#include <vector>
#include <torch/extension.h>

#include "cpu/rotate_nms.h"

#ifdef WITH_CUDA
#include "cuda/rotate_nms_cuda.h"
#endif

#include "nms_methods.h"


// Interface for Python
at::Tensor rotate_nms(const at::Tensor& dets,
               const at::Tensor& scores,
               const float threshold) 
{

  if (dets.type().is_cuda()) {
#ifdef WITH_CUDA
    if (dets.numel() == 0)
      return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
    auto b = at::cat({dets, scores.unsqueeze(1)}, 1);
    return rotate_nms_cuda(b, threshold);
#endif
  }

  at::Tensor result = rotate_nms_cpu(dets, scores, threshold);
  return result;
}

std::tuple<at::Tensor, at::Tensor> rotate_soft_nms(at::Tensor& dets,
                at::Tensor& scores,
                const float nms_thresh=0.3,
                const float sigma=0.5,
                const float score_thresh=0.001,
                const int method=NMS_METHOD::GAUSSIAN) 
{

  if (dets.type().is_cuda()) {
#ifdef WITH_CUDA
    AT_ERROR("Not compiled with GPU support");
#endif
  }

  return rotate_soft_nms_cpu(dets, scores, nms_thresh, sigma, score_thresh, method);
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
