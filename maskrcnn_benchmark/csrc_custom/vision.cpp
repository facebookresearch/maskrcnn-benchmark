#include "ave_dist_loss.h"
#include "hough_voting.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	
  m.def("average_distance_loss_forward", &ave_dist_loss_forward, "average_distance_loss_forward");
  m.def("average_distance_loss_backward", &ave_dist_loss_backward, "average_distance_loss_backward");
  m.def("hough_voting_forward", &hough_voting_forward, "hough_voting_forward");
}
