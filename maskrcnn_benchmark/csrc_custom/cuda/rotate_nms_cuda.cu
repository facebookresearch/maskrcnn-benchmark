
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

//#include <torch/extension.h>

#include <cuda_runtime.h>

#include <vector>
#include <iostream>
#include <stdio.h>

#include <cmath>

#include "rotate_rect_ops.h"

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(unsigned long long) * 8;


#if 1

__device__ inline float devRotateIoU(float const * const region1, float const * const region2) {
  
  float area1 = region1[2] * region1[3];
  float area2 = region2[2] * region2[3];
  float area_inter = inter(region1, region2);

  float iou = area_inter / (area1 + area2 - area_inter + 1e-8); 

  // printf("area1: %.3f, area2: %.3f, area_inter: %.3f, iou: %.3f\n", 
  //     area1, area2, area_inter, iou);
  return iou;
}
#else 
template <typename T>
__device__ inline float devRotateIoU(T const * const region1, T const * const region2) {

  return computeRectIoU(region1, region2);
}
#endif


__global__ void rotate_nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const float *dev_boxes, unsigned long long *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  // cache all column data in this block
  __shared__ float block_boxes[threadsPerBlock * 6];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 6 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 6 + 0];
    block_boxes[threadIdx.x * 6 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 6 + 1];
    block_boxes[threadIdx.x * 6 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 6 + 2];
    block_boxes[threadIdx.x * 6 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 6 + 3];
    block_boxes[threadIdx.x * 6 + 4] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 6 + 4];
    block_boxes[threadIdx.x * 6 + 5] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 6 + 5];
  }
  __syncthreads();

  // iterate across each row in this block
  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;  // current row
    const float *cur_box = dev_boxes + cur_box_idx * 6;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;  // if they are the same, skip to next (column)
    }

    // for this row, calculate all ious with each column
    for (i = start; i < col_size; i++) {
      float iou = devRotateIoU(cur_box, block_boxes + i * 6);
      // printf("iou: %.3f\n", iou);
      if (iou > nms_overlap_thresh) {
        t |= 1ULL << i;  // basically storing all overlaps across the columns, hashed into one single ULL index
      }
    }

    const int col_blocks = THCCeilDiv(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

__global__ void overlaps_kernel(const int N, const int K, const float* dev_boxes,
                           const float * dev_query_boxes, float* dev_overlaps) {

  const int col_start = blockIdx.y;
  const int row_start = blockIdx.x;

  const int row_size =
        min(N - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(K - col_start * threadsPerBlock, threadsPerBlock);


  __shared__ float block_boxes[threadsPerBlock * 5];
  __shared__ float block_query_boxes[threadsPerBlock * 5];
  if (threadIdx.x < col_size) {
    block_query_boxes[threadIdx.x * 5 + 0] =
        dev_query_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 0];
    block_query_boxes[threadIdx.x * 5 + 1] =
        dev_query_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 1];
    block_query_boxes[threadIdx.x * 5 + 2] =
        dev_query_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 2];
    block_query_boxes[threadIdx.x * 5 + 3] =
        dev_query_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 3];
    block_query_boxes[threadIdx.x * 5 + 4] =
        dev_query_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 4];
  }

  if (threadIdx.x < row_size) {
    block_boxes[threadIdx.x * 5 + 0] =
        dev_boxes[(threadsPerBlock * row_start + threadIdx.x) * 5 + 0];
    block_boxes[threadIdx.x * 5 + 1] =
        dev_boxes[(threadsPerBlock * row_start + threadIdx.x) * 5 + 1];
    block_boxes[threadIdx.x * 5 + 2] =
        dev_boxes[(threadsPerBlock * row_start + threadIdx.x) * 5 + 2];
    block_boxes[threadIdx.x * 5 + 3] =
        dev_boxes[(threadsPerBlock * row_start + threadIdx.x) * 5 + 3];
    block_boxes[threadIdx.x * 5 + 4] =
        dev_boxes[(threadsPerBlock * row_start + threadIdx.x) * 5 + 4];
  }

  __syncthreads();

  if (threadIdx.x < row_size) {

    for(int i = 0;i < col_size; i++) {
      int offset = row_start*threadsPerBlock * K + col_start*threadsPerBlock + threadIdx.x*K+ i ;
      dev_overlaps[offset] = devRotateIoU(block_boxes + threadIdx.x * 5, block_query_boxes + i * 5);
    }
  }
}


void _iou_matrix_launcher(float* overlaps, const float* boxes, const float* query_boxes,
        int n, int k, cudaStream_t stream)
{

  dim3 blocks(THCCeilDiv(n, threadsPerBlock),
              THCCeilDiv(k, threadsPerBlock));

  dim3 threads(threadsPerBlock);

  overlaps_kernel<<<blocks, threads, 0, stream>>>(n, k,
                                    boxes,
                                    query_boxes,
                                    overlaps);
  cudaThreadSynchronize();

}


// boxes is a N x 6 tensor
at::Tensor rotate_nms_cuda(const at::Tensor& boxes, float nms_overlap_thresh) {
  using scalar_t = float;
  AT_ASSERTM(boxes.type().is_cuda(), "boxes must be a CUDA tensor");
  auto scores = boxes.select(1, 5);
  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));
  auto boxes_sorted = boxes.index_select(0, order_t);

  int boxes_num = boxes.size(0);

  const int col_blocks = THCCeilDiv(boxes_num, threadsPerBlock);

  scalar_t* boxes_dev = boxes_sorted.data<scalar_t>();

  THCState *state = at::globalContext().lazyInitCUDA(); // TODO replace with getTHCState

  unsigned long long* mask_dev = NULL;
  //THCudaCheck(THCudaMalloc(state, (void**) &mask_dev,
  //                      boxes_num * col_blocks * sizeof(unsigned long long)));

  mask_dev = (unsigned long long*) THCudaMalloc(state, boxes_num * col_blocks * sizeof(unsigned long long));

  dim3 blocks(THCCeilDiv(boxes_num, threadsPerBlock),
              THCCeilDiv(boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);
  rotate_nms_kernel<<<blocks, threads>>>(boxes_num,
                                  nms_overlap_thresh,
                                  boxes_dev,
                                  mask_dev);

  std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
  THCudaCheck(cudaMemcpy(&mask_host[0],
                        mask_dev,
                        sizeof(unsigned long long) * boxes_num * col_blocks,
                        cudaMemcpyDeviceToHost));

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  at::Tensor keep = at::empty({boxes_num}, boxes.options().dtype(at::kLong).device(at::kCPU));
  int64_t* keep_out = keep.data<int64_t>();

  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_out[num_to_keep++] = i;
      unsigned long long *p = &mask_host[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }

  THCudaFree(state, mask_dev);
  // TODO improve this part
  return std::get<0>(order_t.index({
                       keep.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep).to(
                         order_t.device(), keep.scalar_type())
                     }).sort(0, false));
}


at::Tensor rotate_iou_matrix_cuda(
    const at::Tensor& r_boxes1, const at::Tensor& r_boxes2
)
{
  int N = r_boxes1.size(0);
  int M = r_boxes2.size(0);

  at::Tensor iou_matrix = at::zeros({N, M}, r_boxes1.options());

  if (N == 0 || M == 0)
    return iou_matrix;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  _iou_matrix_launcher(iou_matrix.data<float>(), r_boxes1.contiguous().data<float>(),
        r_boxes2.contiguous().data<float>(), N, M, stream);

  THCudaCheck(cudaGetLastError());

  return iou_matrix;
}