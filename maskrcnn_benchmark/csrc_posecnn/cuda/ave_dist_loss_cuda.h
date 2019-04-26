#pragma once
#include <torch/extension.h>

std::vector<at::Tensor> ave_dist_loss_forward_cuda(
    const at::Tensor& poses_pred, const at::Tensor& poses_target, const at::Tensor& poses_labels, const at::Tensor& points, const at::Tensor& symmetry,
    const float margin);

at::Tensor ave_dist_loss_backward_cuda(const at::Tensor& grad, const at::Tensor& bottom_diff);
