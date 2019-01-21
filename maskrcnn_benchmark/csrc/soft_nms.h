// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once
#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif


at::Tensor soft_nms(const at::Tensor& dets,
                    at::Tensor& scores,
                    const float threshold,
                    const float sigma) {

  if (dets.type().is_cuda()) {
#ifdef WITH_CUDA
    AT_ERROR("Soft NMS Does Not have GPU support");
#endif
  }

  at::Tensor result = soft_nms_cpu(dets, scores, threshold, sigma);

  return result;
}
