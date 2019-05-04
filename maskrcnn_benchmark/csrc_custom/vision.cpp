#include "rotate_nms.h"
#include "RROIPool.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  m.def("rotate_nms", &rotate_nms, "rotate_nms");
  m.def("rotate_iou_matrix", &rotate_iou_matrix, "rotate_iou_matrix");

// rotated ROI implementations
  m.def("rotate_roi_pool_forward", &RROIPool_forward, "RROIPool_forward");
  m.def("rotate_roi_pool_backward", &RROIPool_backward, "RROIPool_backward");

}
