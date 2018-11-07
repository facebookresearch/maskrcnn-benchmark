// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <torch/script.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "nms.h"
#include "ROIAlign.h"
#include "ROIPool.h"

using namespace at;

Tensor merge_levels(const Tensor& level_idx, const std::vector<Tensor> levels_data) {
  AT_CHECK(levels_data.size() > 0, "need non-empty levels_data");
  Tensor res = at::zeros({level_idx.size(0), levels_data[0].size(1),
                          levels_data[0].size(2), levels_data[0].size(3)},
                          levels_data[0].options());
  for (int64_t l = 0; l<(int64_t) levels_data.size(); l++) {
    res.masked_scatter_((level_idx==l).view({-1, 1, 1, 1}), levels_data[l]);
  }
  return res;
}

// thin wrapper because we cannot get it from aten in Python due to overloads
Tensor upsample_bilinear(const Tensor& inp, int64_t w, int64_t h) {
  return at::upsample_bilinear2d(inp.unsqueeze(0).unsqueeze(0), {w, h}, false).squeeze(0).squeeze(0);
}

Tensor put_text(const Tensor& inp, int64_t x, int64_t y, const Tensor& color, std::string s) {
  Tensor res = inp.to(kCPU).clone();
  auto c = color.accessor<int64_t, 1>();
  cv::Mat cv_res(inp.size(0), inp.size(1), CV_8UC3, (void*) res.data<uint8_t>());
  putText(cv_res, s, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(c[0], c[1], c[2]), 1);
  return res;
}


static auto registry =
  torch::jit::RegisterOperators()
    .op("maskrcnn_benchmark::nms", &nms)
    .op("maskrcnn_benchmark::roi_align_forward(Tensor input, Tensor rois, float spatial_scale, int pooled_height, int pooled_width, int sampling_ratio) -> Tensor", &ROIAlign_forward)
    .op("maskrcnn_benchmark::merge_levels", &merge_levels)
    .op("maskrcnn_benchmark::put_text", &put_text)
    .op("maskrcnn_benchmark::upsample_bilinear", &upsample_bilinear);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms", &nms, "non-maximum suppression");
  m.def("roi_align_forward", &ROIAlign_forward, "ROIAlign_forward");
  m.def("roi_align_backward", &ROIAlign_backward, "ROIAlign_backward");
  m.def("roi_pool_forward", &ROIPool_forward, "ROIPool_forward");
  m.def("roi_pool_backward", &ROIPool_backward, "ROIPool_backward");
}
