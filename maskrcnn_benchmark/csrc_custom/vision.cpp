#include "rotate_nms.h"
#include "RROIPool.h"
#include "RROIAlign.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  m.def("rotate_nms", &rotate_nms, "rotate_nms");
  m.def("rotate_soft_nms", &rotate_soft_nms, "rotate_soft_nms");
  m.def("rotate_iou_matrix", &rotate_iou_matrix, "rotate_iou_matrix");

// rotated ROI implementations
  m.def("rotate_roi_pool_forward", &RROIPool_forward, "RROIPool_forward");
  m.def("rotate_roi_pool_backward", &RROIPool_backward, "RROIPool_backward");

  m.def("rotate_roi_align_forward", &RROIAlign_forward, "RROIAlign_forward");
  m.def("rotate_roi_align_backward", &RROIAlign_backward, "RROIAlign_backward");

}
