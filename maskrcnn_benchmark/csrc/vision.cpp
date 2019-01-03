// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "nms.h"
#include "ROIAlign.h"
#include "ROIPool.h"
#include "generate_mask_targets.h"
#include "box_iou.h"
#include "box_encode.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms", &nms, "non-maximum suppression");
  m.def("roi_align_forward", &ROIAlign_forward, "ROIAlign_forward");
  m.def("roi_align_backward", &ROIAlign_backward, "ROIAlign_backward");
  m.def("roi_pool_forward", &ROIPool_forward, "ROIPool_forward");
  m.def("roi_pool_backward", &ROIPool_backward, "ROIPool_backward");
  m.def("generate_mask_targets", &generate_mask_targets, "generate_mask_targets");
  m.def("box_iou", &box_iou, "box_iou");
  m.def("box_encode", &box_encode, "box_encode");  
}
