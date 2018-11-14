// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <torch/script.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "nms.h"
#include "ROIAlign.h"

using namespace at;

// thin wrapper because we cannot get it from aten in Python due to overloads
Tensor upsample_bilinear(const Tensor& inp, int64_t w, int64_t h) {
  return at::upsample_bilinear2d(inp.unsqueeze(0).unsqueeze(0), {w, h}, false).squeeze(0).squeeze(0);
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
    .op("maskrcnn_benchmark::add_annotations", &add_annotations)
    .op("maskrcnn_benchmark::upsample_bilinear", &upsample_bilinear);

