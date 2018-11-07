/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 */

#pragma once

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif


at::Tensor compute_flow(const at::Tensor& boxes,
                        const int height,
                        const int width) {
  if (boxes.type().is_cuda()) {
#ifdef WITH_CUDA
    // TODO raise error if not compiled with cuda
    return compute_flow_cuda(boxes, height, width);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}


