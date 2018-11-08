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

// not terribly efficient, but allows us to keep the annotations in the model
Tensor add_annotations(const Tensor& image, const Tensor& labels_, const Tensor& scores_, const Tensor& bboxes_, std::string class_names_string, const Tensor& color_) {
  Tensor res = image.to(kCPU).clone();
  auto color = color_.accessor<int64_t, 1>();
  auto labels = labels_.accessor<int64_t, 1>();
  auto scores = scores_.accessor<float, 1>();
  auto bboxes = bboxes_.accessor<float, 2>();

  std::stringstream sstream(class_names_string);
  std::string s;
  std::vector<std::string> class_names;
  while (std::getline(sstream, s, ',')) {
    class_names.emplace_back(s);
  }

  cv::Mat cv_res(res.size(0), res.size(1), CV_8UC3, (void*) res.data<uint8_t>());
  for (int64_t i = 0; i < labels.size(0); i++) {
    std::stringstream text;
    text.precision(2);
    text << class_names[labels[i]] << ": " << scores[i];
    putText(cv_res, text.str(), cv::Point(bboxes[i][0], bboxes[i][1]), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(color[0], color[1], color[2]), 1);
  }
  return res;
}


static auto registry =
  torch::jit::RegisterOperators()
    .op("maskrcnn_benchmark::nms", &nms)
    .op("maskrcnn_benchmark::roi_align_forward(Tensor input, Tensor rois, float spatial_scale, int pooled_height, int pooled_width, int sampling_ratio) -> Tensor", &ROIAlign_forward)
    .op("maskrcnn_benchmark::merge_levels", &merge_levels)
    .op("maskrcnn_benchmark::put_text", &put_text)
    .op("maskrcnn_benchmark::add_annotations", &add_annotations)
    .op("maskrcnn_benchmark::upsample_bilinear", &upsample_bilinear);


