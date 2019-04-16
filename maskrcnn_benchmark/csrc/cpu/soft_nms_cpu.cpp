// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "cpu/vision.h"


template <typename scalar_t>
std::pair<at::Tensor, at::Tensor> soft_nms_cpu_kernel(const at::Tensor& dets,
                                                      const at::Tensor& scores,
                                                      const float threshold,
                                                      const float sigma) {
  AT_ASSERTM(!dets.type().is_cuda(), "dets must be a CPU tensor");
  AT_ASSERTM(!scores.type().is_cuda(), "scores must be a CPU tensor");
  AT_ASSERTM(dets.type() == scores.type(), "dets should have the same type as scores");

  if (dets.numel() == 0) {
    return std::make_pair(at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU)),
                          at::empty({0}, scores.options().dtype(at::kFloat).device(at::kCPU)));
  }

  auto x1_t = dets.select(1, 0).contiguous();
  auto y1_t = dets.select(1, 1).contiguous();
  auto x2_t = dets.select(1, 2).contiguous();
  auto y2_t = dets.select(1, 3).contiguous();

  auto scores_t = scores.clone();

  at::Tensor areas_t = (x2_t - x1_t + 1) * (y2_t - y1_t + 1);
  auto ndets = dets.size(0);
  auto inds_t = at::arange(ndets, dets.options().dtype(at::kLong).device(at::kCPU));
  
  auto x1 = x1_t.data<scalar_t>();
  auto y1 = y1_t.data<scalar_t>();
  auto x2 = x2_t.data<scalar_t>();
  auto y2 = y2_t.data<scalar_t>();
  auto s = scores_t.data<scalar_t>();
  auto inds = inds_t.data<int64_t>();
  auto areas = areas_t.data<scalar_t>();

  for (int64_t i = 0; i < ndets; i++) {

    auto ix1 = x1[i];
    auto iy1 = y1[i];
    auto ix2 = x2[i];
    auto iy2 = y2[i];
    auto is = s[i];
    auto ii = inds[i];
    auto iarea = areas[i];

    auto maxpos = scores_t.slice(0, i, ndets).argmax().item<int64_t>() + i;

    // add max box as a detection
    x1[i] = x1[maxpos];
    y1[i] = y1[maxpos];
    x2[i] = x2[maxpos];
    y2[i] = y2[maxpos];
    s[i] = s[maxpos];
    inds[i] = inds[maxpos];
    areas[i] = areas[maxpos];

    // swap ith box with position of max box
    x1[maxpos] = ix1;
    y1[maxpos] = iy1;
    x2[maxpos] = ix2;
    y2[maxpos] = iy2;
    s[maxpos] = is;
    inds[maxpos] = ii;
    areas[maxpos] = iarea;

    ix1 = x1[i];
    iy1 = y1[i];
    ix2 = x2[i];
    iy2 = y2[i];
    iarea = areas[i];

    // NMS iterations, note that ndets changes if detection boxes
    // fall below threshold
    for (int64_t j = i + 1; j < ndets; j++) {
      auto xx1 = std::max(ix1, x1[j]);
      auto yy1 = std::max(iy1, y1[j]);
      auto xx2 = std::min(ix2, x2[j]);
      auto yy2 = std::min(iy2, y2[j]);

      auto w = std::max(static_cast<scalar_t>(0), xx2 - xx1 + 1);
      auto h = std::max(static_cast<scalar_t>(0), yy2 - yy1 + 1);

      auto inter = w * h;
      auto ovr = inter / (iarea + areas[j] - inter);

      s[j] = s[j] * std::exp(- std::pow(ovr, 2.0) / sigma);

      // if box score falls below threshold, discard the box by
      // swapping with last box update ndets
      if (s[j] < threshold) {
        x1[j] = x1[ndets - 1];
        y1[j] = y1[ndets - 1];
        x2[j] = x2[ndets - 1];
        y2[j] = y2[ndets - 1];
        s[j] = s[ndets - 1];
        inds[j] = inds[ndets - 1];
        areas[j] = areas[ndets - 1];
        j--;
        ndets--;
      }
    }
  }
  return std::make_pair(inds_t.slice(0, 0, ndets), scores_t.slice(0, 0, ndets));
}

std::pair<at::Tensor, at::Tensor> soft_nms_cpu(const at::Tensor& dets,
                                               const at::Tensor& scores,
                                               const float threshold,
                                               const float sigma) {
  std::pair<at::Tensor, at::Tensor> result;
  AT_DISPATCH_FLOATING_TYPES(dets.type(), "soft_nms", [&] {
    result = soft_nms_cpu_kernel<scalar_t>(dets, scores, threshold, sigma);
  });
  return result;
}
