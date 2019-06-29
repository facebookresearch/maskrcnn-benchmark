// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "cpu/vision.h"

#include "nms_methods.h"


template <typename scalar_t>
at::Tensor nms_cpu_kernel(const at::Tensor& dets,
                          const at::Tensor& scores,
                          const float threshold) {
  AT_ASSERTM(!dets.type().is_cuda(), "dets must be a CPU tensor");
  AT_ASSERTM(!scores.type().is_cuda(), "scores must be a CPU tensor");
  AT_ASSERTM(dets.type() == scores.type(), "dets should have the same type as scores");

  if (dets.numel() == 0) {
    return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
  }

  auto x1_t = dets.select(1, 0).contiguous();
  auto y1_t = dets.select(1, 1).contiguous();
  auto x2_t = dets.select(1, 2).contiguous();
  auto y2_t = dets.select(1, 3).contiguous();

  at::Tensor areas_t = (x2_t - x1_t + 1) * (y2_t - y1_t + 1);

  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));

  auto ndets = dets.size(0);
  at::Tensor suppressed_t = at::zeros({ndets}, dets.options().dtype(at::kByte).device(at::kCPU));

  auto suppressed = suppressed_t.data<uint8_t>();
  auto order = order_t.data<int64_t>();
  auto x1 = x1_t.data<scalar_t>();
  auto y1 = y1_t.data<scalar_t>();
  auto x2 = x2_t.data<scalar_t>();
  auto y2 = y2_t.data<scalar_t>();
  auto areas = areas_t.data<scalar_t>();

  for (int64_t _i = 0; _i < ndets; _i++) {
    auto i = order[_i];
    if (suppressed[i] == 1)
      continue;
    auto ix1 = x1[i];
    auto iy1 = y1[i];
    auto ix2 = x2[i];
    auto iy2 = y2[i];
    auto iarea = areas[i];

    for (int64_t _j = _i + 1; _j < ndets; _j++) {
      auto j = order[_j];
      if (suppressed[j] == 1)
        continue;
      auto xx1 = std::max(ix1, x1[j]);
      auto yy1 = std::max(iy1, y1[j]);
      auto xx2 = std::min(ix2, x2[j]);
      auto yy2 = std::min(iy2, y2[j]);

      auto w = std::max(static_cast<scalar_t>(0), xx2 - xx1 + 1);
      auto h = std::max(static_cast<scalar_t>(0), yy2 - yy1 + 1);
      auto inter = w * h;
      auto ovr = inter / (iarea + areas[j] - inter);
      if (ovr >= threshold)
        suppressed[j] = 1;
   }
  }
  return at::nonzero(suppressed_t == 0).squeeze(1);
}

at::Tensor nms_cpu(const at::Tensor& dets,
               const at::Tensor& scores,
               const float threshold) {
  at::Tensor result;
  AT_DISPATCH_FLOATING_TYPES(dets.type(), "nms", [&] {
    result = nms_cpu_kernel<scalar_t>(dets, scores, threshold);
  });
  return result;
}


template <typename scalar_t>
at::Tensor soft_nms_cpu_kernel(at::Tensor& dets,
                          at::Tensor& scores,
                          at::Tensor& indices,
                          const float nms_thresh,
                          const float sigma,
                          const float score_thresh,
                          const int method
                          ) 
{
  AT_ASSERTM(!dets.type().is_cuda(), "dets must be a CPU tensor");
  AT_ASSERTM(!scores.type().is_cuda(), "scores must be a CPU tensor");
  AT_ASSERTM(dets.type() == scores.type(), "dets should have the same type as scores");

  if (dets.numel() == 0) {
    return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
  }

  auto scores_d = scores.contiguous().data<scalar_t>();
  auto dets_d = dets.contiguous().data<scalar_t>();
  auto indices_d = indices.contiguous().data<int64_t>();

  auto ndets = dets.size(0);

  for (int64_t i = 0; i < ndets; i++) {
    int64_t pos = i + 1;
    auto rem_scores = scores.narrow(/*dim=*/0, /*start=*/pos, /*length=*/ndets - pos);
    // auto order_t = std::get<1>(rem_scores.sort(0, /* descending=*/true));
    // auto order = order_t.data<int64_t>();
    int64_t maxpos = 0;
    scalar_t maxscore = scores_d[ndets - 1];
    if (i != ndets - 1)
    {
      maxpos = rem_scores.argmax().data<int64_t>()[0];
      maxscore = rem_scores.data<scalar_t>()[maxpos];
    }
    if (scores_d[i] < maxscore)
    {
      scalar_t* dd = dets_d + (i*4);
      scalar_t* dd_max = dets_d + ((maxpos+pos)*4);
      scalar_t tmp;
      for (size_t n = 0; n < 4; n++)
      {
        tmp = dd[n];
        dd[n] = dd_max[n];
        dd_max[n] = tmp;
      }
      tmp = scores_d[i];
      scores_d[i] = maxscore;
      scores_d[maxpos+pos] = tmp;

      int64_t tmp_i = indices_d[i];
      indices_d[i] = indices_d[maxpos + pos];
      indices_d[maxpos + pos] = tmp_i;
    }
  
    const scalar_t* bbox = dets_d + (i*4);
    auto ix1 = bbox[0];
    auto iy1 = bbox[1];
    auto ix2 = bbox[2];
    auto iy2 = bbox[3];
    auto iarea = (ix2 - ix1 + 1) * (iy2 - iy1 + 1);

    for (int64_t j = pos; j < ndets; j++) {

      const scalar_t* bbox2 = dets_d + (j*4);
      auto xx1 = std::max(ix1, bbox2[0]);
      auto yy1 = std::max(iy1, bbox2[1]);
      auto xx2 = std::min(ix2, bbox2[2]);
      auto yy2 = std::min(iy2, bbox2[3]);

      auto w = std::max(static_cast<scalar_t>(0), xx2 - xx1 + 1);
      auto h = std::max(static_cast<scalar_t>(0), yy2 - yy1 + 1);
      auto inter = w * h;
      auto jarea = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1);
      auto ovr = inter / (iarea + jarea - inter);

      float weight = 1.0f;
      if (method == NMS_METHOD::GAUSSIAN)
      {
        weight = exp(-(ovr * ovr) / sigma);
      } else if (ovr >= nms_thresh)
      {
        weight = method == NMS_METHOD::LINEAR ? weight - ovr : 0.0f;
      }
      auto& score_j = scores_d[j];
      score_j *= weight;
   }
  }
  return at::nonzero(scores >= score_thresh).squeeze(1);
}

std::tuple<at::Tensor, at::Tensor> soft_nms_cpu(at::Tensor& dets,
               at::Tensor& scores,
               const float nms_thresh,
               const float sigma,
               const float score_thresh,
               const int method
               ) 
{
  auto N = dets.size(0);
  at::Tensor keep;
  at::Tensor indices = at::arange(N, dets.options().dtype(at::kLong).device(at::kCPU));
  if (method == NMS_METHOD::LINEAR || method == NMS_METHOD::GAUSSIAN)
  {
    AT_DISPATCH_FLOATING_TYPES(dets.type(), "soft_nms", [&] {
      keep = soft_nms_cpu_kernel<scalar_t>(dets, scores, indices, nms_thresh, sigma, score_thresh, method);
    });
  } else {
    // original nms
    AT_DISPATCH_FLOATING_TYPES(dets.type(), "nms", [&] {
      keep = nms_cpu_kernel<scalar_t>(dets, scores, nms_thresh);
    });
  }

  return std::make_tuple(indices, keep);
}