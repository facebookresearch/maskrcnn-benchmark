#include "ave_dist_loss.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	
  m.def("average_distance_loss_forward", &ave_dist_loss_forward, "average_distance_loss_forward");
  m.def("average_distance_loss_backward", &ave_dist_loss_backward, "average_distance_loss_backward");
}
