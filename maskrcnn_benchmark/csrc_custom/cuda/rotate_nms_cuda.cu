
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
__device__ inline float trangle_area(float * a, float * b, float * c) {
  return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0]))/2.0;
}

__device__ inline float area(float * int_pts, int num_of_inter) {

  float area = 0.0;
  for(int i = 0;i < num_of_inter - 2;i++) {
    area += fabs(trangle_area(int_pts, int_pts + 2 * i + 2, int_pts + 2 * i + 4));
  }
  return area;
}

__device__ inline void reorder_pts(float * int_pts, int num_of_inter) {



  if(num_of_inter > 0) {
    
    float center[2];
    
    center[0] = 0.0;
    center[1] = 0.0;

    for(int i = 0;i < num_of_inter;i++) {
      center[0] += int_pts[2 * i];
      center[1] += int_pts[2 * i + 1];
    }
    center[0] /= num_of_inter;
    center[1] /= num_of_inter;

    float vs[16];
    float v[2];
    float d;
    for(int i = 0;i < num_of_inter;i++) {
      v[0] = int_pts[2 * i]-center[0];
      v[1] = int_pts[2 * i + 1]-center[1];
      d = sqrt(v[0] * v[0] + v[1] * v[1]);
      v[0] = v[0] / d;
      v[1] = v[1] / d;
      if(v[1] < 0) {
        v[0]= - 2 - v[0];
      }
      vs[i] = v[0];
    }
    
    float temp,tx,ty;
    int j;
    for(int i=1;i<num_of_inter;++i){
      if(vs[i-1]>vs[i]){
        temp = vs[i];
        tx = int_pts[2*i];
        ty = int_pts[2*i+1];
        j=i;
        while(j>0&&vs[j-1]>temp){
          vs[j] = vs[j-1];
          int_pts[j*2] = int_pts[j*2-2];
          int_pts[j*2+1] = int_pts[j*2-1];
          j--;
        }
        vs[j] = temp;
        int_pts[j*2] = tx;
        int_pts[j*2+1] = ty;
      }
    }
  }

}
__device__ inline bool inter2line(float * pts1, float *pts2, int i, int j, float * temp_pts) {

  float a[2];
  float b[2];
  float c[2];
  float d[2];

  float area_abc, area_abd, area_cda, area_cdb;

  a[0] = pts1[2 * i];
  a[1] = pts1[2 * i + 1];

  b[0] = pts1[2 * ((i + 1) % 4)];
  b[1] = pts1[2 * ((i + 1) % 4) + 1];

  c[0] = pts2[2 * j];
  c[1] = pts2[2 * j + 1];

  d[0] = pts2[2 * ((j + 1) % 4)];
  d[1] = pts2[2 * ((j + 1) % 4) + 1];

  area_abc = trangle_area(a, b, c);
  area_abd = trangle_area(a, b, d);
  
  if(area_abc * area_abd >= 0) {
    return false;
  }
  
  area_cda = trangle_area(c, d, a); 
  area_cdb = area_cda + area_abc - area_abd;

  if (area_cda * area_cdb >= 0) {
    return false;
  }
  float t = area_cda / (area_abd - area_abc);      
    
  float dx = t * (b[0] - a[0]);
  float dy = t * (b[1] - a[1]);
  temp_pts[0] = a[0] + dx;
  temp_pts[1] = a[1] + dy;

  return true;
}

__device__ inline bool in_rect(float pt_x, float pt_y, float * pts) {
  
  float ab[2];
  float ad[2];
  float ap[2];

  float abab;
  float abap;
  float adad;
  float adap;

  ab[0] = pts[2] - pts[0];
  ab[1] = pts[3] - pts[1];

  ad[0] = pts[6] - pts[0];
  ad[1] = pts[7] - pts[1];

  ap[0] = pt_x - pts[0];
  ap[1] = pt_y - pts[1];

  abab = ab[0] * ab[0] + ab[1] * ab[1];
  abap = ab[0] * ap[0] + ab[1] * ap[1];
  adad = ad[0] * ad[0] + ad[1] * ad[1];
  adap = ad[0] * ap[0] + ad[1] * ap[1];

  return abab >= abap && abap >= 0 && adad >= adap && adap >= 0;
}

__device__ inline int inter_pts(float * pts1, float * pts2, float * int_pts) {

  int num_of_inter = 0;

  for(int i = 0;i < 4;i++) {
    if(in_rect(pts1[2 * i], pts1[2 * i + 1], pts2)) {
      int_pts[num_of_inter * 2] = pts1[2 * i];
      int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1];
      num_of_inter++;
    }
     if(in_rect(pts2[2 * i], pts2[2 * i + 1], pts1)) {
      int_pts[num_of_inter * 2] = pts2[2 * i];
      int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1];
      num_of_inter++;
    }   
  }

  float temp_pts[2];

  for(int i = 0;i < 4;i++) {
    for(int j = 0;j < 4;j++) {
      bool has_pts = inter2line(pts1, pts2, i, j, temp_pts);
      if(has_pts) {
        int_pts[num_of_inter * 2] = temp_pts[0];
        int_pts[num_of_inter * 2 + 1] = temp_pts[1];
        num_of_inter++;
      }
    }
  }

  return num_of_inter;
}

__device__ inline float inter(float const * const region1, float const * const region2) {

  float pts1[8];
  float pts2[8];
  float int_pts[16];
  int num_of_inter;

  convert_region_to_pts(region1, pts1);
  convert_region_to_pts(region2, pts2);

  num_of_inter = inter_pts(pts1, pts2, int_pts);
  // printf("num_of_inter: %d\n", num_of_inter);

  reorder_pts(int_pts, num_of_inter);

  return area(int_pts, num_of_inter);
}

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
  __shared__ float block_boxes[threadsPerBlock * 5];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 5 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 0];
    block_boxes[threadIdx.x * 5 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 1];
    block_boxes[threadIdx.x * 5 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 2];
    block_boxes[threadIdx.x * 5 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 3];
    block_boxes[threadIdx.x * 5 + 4] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 4];
  }
  __syncthreads();

  // iterate across each row in this block
  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;  // current row
    const float *cur_box = dev_boxes + cur_box_idx * 5;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;  // if they are the same, skip to next (column)
    }

    // for this row, calculate all ious with each column
    for (i = start; i < col_size; i++) {
      float iou = devRotateIoU(cur_box, block_boxes + i * 5);
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

void _rotate_nms_launcher(int64_t* keep_out, int* num_out, const float* boxes, int boxes_num,
          int boxes_dim, float nms_overlap_thresh, cudaStream_t stream)
{
  /**
  Inputs:
  boxes: N,5  (xc,yc,w,h,angle)  ASSUMES already sorted
  boxes_num: N
  boxes_dim: 5
  nms_overlap_thresh: 0-1 e.g. 0.7

  Outputs:
  keep_out: N  (i.e. stores indices of valid boxes)
  num_out: total count of valid indices

  */

  const int col_blocks = THCCeilDiv(boxes_num, threadsPerBlock);

  unsigned long long* mask_dev = NULL;
  // Get the IoUs between each element in the array (N**2 operation)
  // then store all the overlap results in the N*col_blocks array (mask_dev).
  // col_blocks represents the total number of column blocks (blockDim.x) made for the kernel computation
  // Each column block will store a hash of the iou overlaps between each column and row in the block. The hash is a ULL of bit overlaps between one row and all columns in the block
  // then copy the results to host code
  // Each result row is a col_block array, which contains all the iou overlap bool (as a hash) per column block.
  // Loop through the col_block array to aggregate all iou overlap results for that row
  CUDA_CHECK(cudaMalloc(&mask_dev,
                        boxes_num * col_blocks * sizeof(unsigned long long)));

  dim3 blocks(col_blocks, col_blocks);
  dim3 threads(threadsPerBlock);

  rotate_nms_kernel<<<blocks, threads, 0, stream>>>(boxes_num,
                                  nms_overlap_thresh,
                                  boxes,
                                  mask_dev);
  cudaThreadSynchronize();

  std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
  CUDA_CHECK(cudaMemcpy(&mask_host[0],
                        mask_dev,
                        sizeof(unsigned long long) * boxes_num * col_blocks,
                        cudaMemcpyDeviceToHost));

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;  // get column block
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {  // if not zero i.e. no overlap
      keep_out[num_to_keep++] = i;
      unsigned long long *p = &mask_host[0] + i * col_blocks;

      // Loop through the col_block array to aggregate all iou overlap results for that box
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }
  *num_out = num_to_keep;

  CUDA_CHECK(cudaFree(mask_dev));
}


at::Tensor rotate_nms_cuda(
    const at::Tensor& r_boxes, const float nms_threshold, const int max_output
)
{
  int boxes_num = r_boxes.size(0);
  int channels = r_boxes.size(1);

  at::Tensor keep = at::zeros({boxes_num}, r_boxes.options().dtype(at::kLong).device(at::kCPU));

  if (boxes_num == 0)
    return keep;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int num_to_keep = 0;
  _rotate_nms_launcher(keep.data<int64_t>(), &num_to_keep, r_boxes.contiguous().data<float>(), boxes_num,
          channels, nms_threshold, stream);
//  AveragedistanceBackwardLaucher(
//    grad.contiguous().data<float>(), bottom_diff.contiguous().data<float>(),
//    batch_size, channels, output.data<float>(), stream
//  );
  THCudaCheck(cudaGetLastError());

  if (max_output >= 0)
    num_to_keep = std::min(num_to_keep, max_output);

//  printf("GPU: num_to_keep: %d\n", num_to_keep);
  return keep.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep).to(
      r_boxes.device(), keep.scalar_type());

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