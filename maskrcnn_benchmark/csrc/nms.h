// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once
#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif

#include "nms_methods.h"

at::Tensor nms(const at::Tensor& dets,
               const at::Tensor& scores,
               const float threshold) {

  if (dets.type().is_cuda()) {
#ifdef WITH_CUDA
    // TODO raise error if not compiled with CUDA
    if (dets.numel() == 0)
      return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
    auto b = at::cat({dets, scores.unsqueeze(1)}, 1);
    return nms_cuda(b, threshold);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }

  at::Tensor result = nms_cpu(dets, scores, threshold);
  return result;
}

std::tuple<at::Tensor, at::Tensor> soft_nms(at::Tensor& dets,
                at::Tensor& scores,
                const float nms_thresh=0.3,
                const float sigma=0.5,
                const float score_thresh=0.001,
                const int method=NMS_METHOD::GAUSSIAN) 
{

  if (dets.type().is_cuda()) {
#ifdef WITH_CUDA
//     if (dets.numel() == 0)
//       return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
//     return soft_nms_cuda(dets, scores, nms_thresh, sigma, score_thresh, method);
// #else
    AT_ERROR("Not compiled with GPU support");
#endif
  }

  return soft_nms_cpu(dets, scores, nms_thresh, sigma, score_thresh, method);
}
