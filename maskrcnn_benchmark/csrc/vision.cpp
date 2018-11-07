/**
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 */

#include "nms.h"
#include "ROIAlign.h"
#include "ROIPool.h"
#include "compute_flow.h"
#include "generate_mask_targets.h"
#include "match_proposals.h"
#include "box_encode.h"
#include "box_iou.h"
#include "anchor_generator.h"
#include "cuda/generate_proposals.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms", &nms, "non-maximum suppression");
  m.def("roi_align_forward", &ROIAlign_forward, "ROIAlign_forward");
  m.def("roi_align_backward", &ROIAlign_backward, "ROIAlign_backward");
  m.def("roi_pool_forward", &ROIPool_forward, "ROIPool_forward");
  m.def("roi_pool_backward", &ROIPool_backward, "ROIPool_backward");
  m.def("compute_flow", &compute_flow, "compute_flow");
  m.def("generate_mask_targets", &generate_mask_targets, "generate_mask_targets");
  m.def("match_proposals", &match_proposals, "match_proposals");
  m.def("box_encode", &box_encode, "box_encode");
  m.def("box_iou", &box_iou, "box_iou");
  m.def("GeneratePreNMSUprightBoxes", &GeneratePreNMSUprightBoxes, "GeneratePreNMSUprightBoxes");
  m.def("nms_gpu_upright", &utils::nms_gpu_upright, "nms_gpu_upright");
  m.def("anchor_generator", &anchor_generator, "anchor_generator");
}
