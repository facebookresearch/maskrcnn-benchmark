#include "cpu/rotate_nms.h"

#include "rotate_rect_ops.h"


template <typename scalar_t>
at::Tensor rotate_nms_cpu_kernel(const at::Tensor& dets,
                          const float threshold, const int max_output)
{
  AT_ASSERTM(!dets.type().is_cuda(), "dets must be a CPU tensor");

  if (dets.numel() == 0) {
    return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
  }

  auto ndets = dets.size(0);
  at::Tensor suppressed_t = at::zeros({ndets}, dets.options().dtype(at::kByte).device(at::kCPU));

  auto xc_t = dets.select(1, 0).contiguous();
  auto yc_t = dets.select(1, 1).contiguous();
  auto w_t = dets.select(1, 2).contiguous();
  auto h_t = dets.select(1, 3).contiguous();
  auto angle_t = dets.select(1, 4).contiguous();

  auto suppressed = suppressed_t.data<uint8_t>();
  auto xc = xc_t.data<scalar_t>();
  auto yc = yc_t.data<scalar_t>();
  auto w = w_t.data<scalar_t>();
  auto h = h_t.data<scalar_t>();
  auto angle = angle_t.data<scalar_t>();

  // auto areas = areas_t.data<scalar_t>();
  bool limit_output = max_output >= 0;

  scalar_t rect_1[5];
  int num_to_keep = 0;
  for (int64_t i = 0; i < ndets; i++) {
    if (suppressed[i] == 1)
      continue;
    ++num_to_keep;
    if (limit_output && num_to_keep >= max_output)
      break;

    rect_1[0] = xc[i];
    rect_1[1] = yc[i];
    rect_1[2] = w[i];
    rect_1[3] = h[i];
    rect_1[4] = angle[i];

    scalar_t rect_2[5];
    for (int64_t j = i + 1; j < ndets; j++) {
      if (suppressed[j] == 1)
        continue;

      rect_2[0] = xc[j];
      rect_2[1] = yc[j];
      rect_2[2] = w[j];
      rect_2[3] = h[j];
      rect_2[4] = angle[j];

      float ovr = computeRectIoU(rect_1, rect_2);
      if (ovr >= threshold)
        suppressed[j] = 1;
   }
  }
  return at::nonzero(suppressed_t == 0).squeeze(1).narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep);
}

template <typename scalar_t>
void rotate_iou_cpu_kernel(const at::Tensor& r_boxes1,
                          const at::Tensor& r_boxes2,
                          at::Tensor& iou_matrix)
{
  AT_ASSERTM(!r_boxes1.type().is_cuda(), "r_boxes1 must be a CPU tensor");
  AT_ASSERTM(!r_boxes2.type().is_cuda(), "r_boxes2 must be a CPU tensor");

  if (r_boxes1.numel() == 0 || r_boxes2.numel() == 0)
    return;

  int N = r_boxes1.size(0);
  int M = r_boxes2.size(0);

  // r_boxes1 data
  auto xc_t1 = r_boxes1.select(1, 0).contiguous();
  auto yc_t1 = r_boxes1.select(1, 1).contiguous();
  auto w_t1 = r_boxes1.select(1, 2).contiguous();
  auto h_t1 = r_boxes1.select(1, 3).contiguous();
  auto angle_t1 = r_boxes1.select(1, 4).contiguous();

  auto xc1 = xc_t1.data<scalar_t>();
  auto yc1 = yc_t1.data<scalar_t>();
  auto w1 = w_t1.data<scalar_t>();
  auto h1 = h_t1.data<scalar_t>();
  auto angle1 = angle_t1.data<scalar_t>();

  // r_boxes2 data
  auto xc_t2 = r_boxes2.select(1, 0).contiguous();
  auto yc_t2 = r_boxes2.select(1, 1).contiguous();
  auto w_t2 = r_boxes2.select(1, 2).contiguous();
  auto h_t2 = r_boxes2.select(1, 3).contiguous();
  auto angle_t2 = r_boxes2.select(1, 4).contiguous();

  auto xc2 = xc_t2.data<scalar_t>();
  auto yc2 = yc_t2.data<scalar_t>();
  auto w2 = w_t2.data<scalar_t>();
  auto h2 = h_t2.data<scalar_t>();
  auto angle2 = angle_t2.data<scalar_t>();

  auto iou_data = iou_matrix.data<scalar_t>();

  scalar_t rect_1[5];
  for (int64_t i = 0; i < N; i++)
  {
    rect_1[0] = xc1[i];
    rect_1[1] = yc1[i];
    rect_1[2] = w1[i];
    rect_1[3] = h1[i];
    rect_1[4] = angle1[i];

    scalar_t rect_2[5];
    for (int64_t j = 0; j < M; j++)
    {
      rect_2[0] = xc2[j];
      rect_2[1] = yc2[j];
      rect_2[2] = w2[j];
      rect_2[3] = h2[j];
      rect_2[4] = angle2[j];

      float iou = computeRectIoU(rect_1, rect_2);
      iou_data[i*M+j] = iou;
    }
  }
}

at::Tensor rotate_nms_cpu(const at::Tensor& r_boxes,
                   const float nms_threshold, const int max_output)
{
  at::Tensor result;
  AT_DISPATCH_FLOATING_TYPES(r_boxes.type(), "rotate_nms", [&] {
    result = rotate_nms_cpu_kernel<scalar_t>(r_boxes, nms_threshold, max_output);
  });
  return result;
}

at::Tensor rotate_iou_matrix_cpu(const at::Tensor& r_boxes1,
                   const at::Tensor& r_boxes2)
{
  int N = r_boxes1.size(0);
  int M = r_boxes2.size(0);

  at::Tensor iou_matrix = at::zeros({N, M}, r_boxes1.options());

  if (N == 0 || M == 0)
    return iou_matrix;

  AT_DISPATCH_FLOATING_TYPES(r_boxes1.type(), "rotate_iou", [&] {
    rotate_iou_cpu_kernel<scalar_t>(r_boxes1, r_boxes2, iou_matrix);
  });
  return iou_matrix;
}