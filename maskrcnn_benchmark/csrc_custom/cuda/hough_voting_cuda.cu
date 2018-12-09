#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <math.h>
#include <float.h>

#include <thrust/execution_policy.h>  // for certain cuda versions, this is where 'thrust::device' is

// #include <thrust/device_vector.h>
// #include <thrust/copy.h>
#include <thrust/extrema.h>
#include <cuda_runtime.h>

#include <iostream>

// #include "hough_voting_cuda_utils.h"


#define THREADS_PER_BLOCK 512
#define VERTEX_CHANNELS 3 
#define MAX_ROI 64

#define PRINT(a) std::cout << #a << ": " << a << std::endl;

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)



__device__ inline float angle_distance(int cx, int cy, int x, int y, float u, float v)
{
  float dx = cx - x;
  float dy = cy - y;
  float n1 = sqrt(u * u + v * v);
  float n2 = sqrt(dx * dx + dy * dy);
  float dot = u * dx + v * dy;
  float distance = dot / (n1 * n2);

  return distance;
}

__device__ inline void project_box(int cls, const float* extents, const float* meta_data, float distance, float factor, float* threshold)
{
  float xHalf = extents[cls * 3 + 0] * 0.5;
  float yHalf = extents[cls * 3 + 1] * 0.5;
  float zHalf = extents[cls * 3 + 2] * 0.5;
  float bb3D[24];

  bb3D[0] = xHalf; bb3D[1] = yHalf; bb3D[2] = zHalf + distance;
  bb3D[3] = -xHalf; bb3D[4] = yHalf; bb3D[5] = zHalf + distance;
  bb3D[6] = xHalf; bb3D[7] = -yHalf; bb3D[8] = zHalf + distance;
  bb3D[9] = -xHalf; bb3D[10] = -yHalf; bb3D[11] = zHalf + distance;
  bb3D[12] = xHalf; bb3D[13] = yHalf; bb3D[14] = -zHalf + distance;
  bb3D[15] = -xHalf; bb3D[16] = yHalf; bb3D[17] = -zHalf + distance;
  bb3D[18] = xHalf; bb3D[19] = -yHalf; bb3D[20] = -zHalf + distance;
  bb3D[21] = -xHalf; bb3D[22] = -yHalf; bb3D[23] = -zHalf + distance;

  float fx = meta_data[0];
  float fy = meta_data[4];
  float px = meta_data[2];
  float py = meta_data[5];
  float minX = 1e8;
  float maxX = -1e8;
  float minY = 1e8;
  float maxY = -1e8;
  for (int i = 0; i < 8; i++)
  {
    float x = fx * (bb3D[i * 3] / bb3D[i * 3 + 2])  + px;
    float y = fy * (bb3D[i * 3 + 1] / bb3D[i * 3 + 2])  + py;
    minX = fmin(minX, x);
    minY = fmin(minY, y);
    maxX = fmax(maxX, x);
    maxY = fmax(maxY, y);
  }
  float width = maxX - minX + 1;
  float height = maxY - minY + 1;
  *threshold = fmax(width, height) * factor;
}



__global__ void compute_arrays_kernel(const int nthreads, const int* labelmap,
    int* arrays, int* array_size, const int height, const int width) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    int n = index / (height * width);
    int mask = labelmap[index];
    if (mask > 0)
    {
      int size = atomicAdd(array_size + n, 1);
      int offset = n * height * width + size;
      arrays[offset] = index % (height * width);
    }
  }
}

__global__ void compute_hough_kernel(const int nthreads, float* hough_space, float* hough_data, 
    const float* vertmap, const float* extents, const float* meta_data, const int* arrays, const int* array_size, 
    const int* class_indexes, const int height, const int width, const float inlierThreshold, const int skip_pixels) 
{
  __shared__ float s_meta_data[9];

  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    if (threadIdx.x == 0)
    {
      for (int i = 0; i < 9; ++i)
      {
        s_meta_data[i] = meta_data[i]; 
      }
    }
    __syncthreads();

    // (cls, cx, cy) is an element in the hough space
    int n = index / (height * width);

    int cls = class_indexes[n];
    int pix = index % (height * width);
    int cx = pix % width;
    int cy = pix / width;
    int size = array_size[n];
    float distance = 0;
    float threshold = 0;

    for (int i = 0; i < size; i += skip_pixels)
    {
      int offset = n * height * width + i;
      int location = arrays[offset];  // H * W
      int x = location % width;
      int y = location / (width);

      // read the direction
      offset = n * height * width * VERTEX_CHANNELS + (y * width + x) * VERTEX_CHANNELS;
      float u = vertmap[offset];
      float v = vertmap[offset + 1];
      float d = exp(vertmap[offset + 2]);

      // vote
      if (angle_distance(cx, cy, x, y, u, v) > inlierThreshold)
      {
        project_box(cls, extents, s_meta_data, d, 0.6, &threshold);
        float dx = fabsf(x - cx);
        float dy = fabsf(y - cy);
        if (dx < threshold && dy < threshold)
        {
          hough_space[index]++;
          distance += d;
        }
      }
    }

    if (hough_space[index] > 0)
    {
      distance /= hough_space[index];

      float bb_width = -1;
      float bb_height = -1;
      for (int i = 0; i < size; i += skip_pixels)
      {
        int offset = n * height * width + i;
        int location = arrays[offset];
        int x = location % width;
        int y = location / width;

        // read the direction
        offset = n * height * width * VERTEX_CHANNELS + (y * width + x) * VERTEX_CHANNELS;
        float u = vertmap[offset];
        float v = vertmap[offset + 1];

        // vote
        if (angle_distance(cx, cy, x, y, u, v) > inlierThreshold)
        {
          project_box(cls, extents, s_meta_data, distance, 0.6, &threshold);
          float dx = fabsf(x - cx);
          float dy = fabsf(y - cy);
          if (dx > bb_width && dx < threshold && dy < threshold)
            bb_width = dx;
          if (dy > bb_height && dx < threshold && dy < threshold)
            bb_height = dy;
        }
      }

      int offset = n * height * width * 3 + 3 * (cy * width + cx);
      hough_data[offset] = distance;
      hough_data[offset + 1] = 2 * bb_height;
      hough_data[offset + 2] = 2 * bb_width;
    }
  }
}

__global__ void compute_rois_kernel(const int nthreads, float* top_box, float* top_pose, 
    const float* poses, const float* meta_data, const float* hough_space, const float* hough_data, const int* max_indexes, const int* class_indexes,
    const int height, const int width) 
{
  __shared__ float s_f[4];

  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    if (threadIdx.x == 0)
    {
      s_f[0] = meta_data[0]; // fx
      s_f[1] = meta_data[4]; // fy
      s_f[2] = meta_data[2]; // px
      s_f[3] = meta_data[5]; // py
    }
    __syncthreads();

    float fx = s_f[0];
    float fy = s_f[1];
    float px = s_f[2];
    float py = s_f[3];

    float scale = 0.05;
    int max_index = max_indexes[index];
    float max_hs_idx = hough_space[max_index];
    int ind = max_index / (height * width);
    int cls = class_indexes[ind];
    int n = max_index % (height * width);
    int x = n % width;
    int y = n / width;

    float rx = (x - px) / fx;
    float ry = (y - py) / fy;

    int offset = ind * height * width * 3 + 3 * (y * width + x);
    float bb_distance = hough_data[offset];
    float bb_height = hough_data[offset + 1];
    float bb_width = hough_data[offset + 2];

    int roi_index = index; 
    top_box[roi_index * 7 + 0] = 0;
    top_box[roi_index * 7 + 1] = cls;
    top_box[roi_index * 7 + 2] = x - bb_width * (0.5 + scale);
    top_box[roi_index * 7 + 3] = y - bb_height * (0.5 + scale);
    top_box[roi_index * 7 + 4] = x + bb_width * (0.5 + scale);
    top_box[roi_index * 7 + 5] = y + bb_height * (0.5 + scale);
    top_box[roi_index * 7 + 6] = max_hs_idx;
    
    top_pose[roi_index * 7 + 0] = poses[index*4];
    top_pose[roi_index * 7 + 1] = poses[index*4+1];
    top_pose[roi_index * 7 + 2] = poses[index*4+2];
    top_pose[roi_index * 7 + 3] = poses[index*4+3];
    top_pose[roi_index * 7 + 4] = rx * bb_distance;
    top_pose[roi_index * 7 + 5] = ry * bb_distance;
    top_pose[roi_index * 7 + 6] = bb_distance;

  }
}

int HoughVotingForwardLaucher(
    const int* labels, const int* labelmap, const float* vertmap, const float* extents, const float* meta_data, const float* poses,
    const int batch_size, const int height, const int width, 
    const float inlierThreshold,// const float votingThreshold, const float perThreshold, 
    const int skip_pixels, 
    float* top_box, float* top_pose, cudaStream_t stream)
{
  const int kThreadsPerBlock = THREADS_PER_BLOCK;
  cudaError_t err;

  const int N = batch_size;

  // step 1: compute a label index array for each instance
  int dims = N * height * width;
  int* arrays;// = arrays_vec.get();
  cudaMalloc((void **)&arrays, dims * sizeof(int));
  cudaMemset(arrays, 0, N * sizeof(int));

  int* array_sizes;// = array_sizes_vec.get();  
  cudaMalloc((void **)&array_sizes, N * sizeof(int));
  cudaMemset(array_sizes, 0, N * sizeof(int));

  int output_size = N * height * width;
  compute_arrays_kernel<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, stream>>>(
      output_size, labelmap, arrays, array_sizes, height, width);
  cudaThreadSynchronize();

  // DEBUG
  // std::vector<int> array_sizes_host(N);
  // cudaMemcpy(&array_sizes_host[0], array_sizes, N * sizeof(int), cudaMemcpyDeviceToHost);
  // std::vector<int> labels_host(N);
  // cudaMemcpy(&labels_host[0], labels, N * sizeof(int), cudaMemcpyDeviceToHost);
  // std::vector<int> arrays_host(N*height*width);
  // cudaMemcpy(&arrays_host[0], arrays, N*height*width * sizeof(int), cudaMemcpyDeviceToHost);
  // // std::vector<float> meta_data_host(9);
  // // cudaMemcpy(&meta_data_host[0], meta_data, 9 * sizeof(float), cudaMemcpyDeviceToHost);
  // for (int n = 0; n < N; n++)
  // {
  //   printf("Class %d) %d) (labels count: %d), sample value: %d\n", labels_host[n], n, array_sizes_host[n], arrays_host[n*height*width]); 
  // }
  // // for (int n = 0; n < 9; n++)
  // // {
  // //   printf("META: %.3f\n", meta_data_host[n]);
  // // }
  // // printf("\n");
  // // 

  // step 2: compute the hough space
  float* hough_space; // = thrust::raw_pointer_cast(hough_space_vec.data());
  cudaMalloc((void **)&hough_space, N * height * width * sizeof(float));
  if (cudaMemset(hough_space, 0, N * height * width * sizeof(float)) != cudaSuccess)
    fprintf(stderr, "reset error\n");

  float* hough_data; // = thrust::raw_pointer_cast(hough_data_vec.data());
  cudaMalloc((void **)&hough_data, N * height * width * 3 * sizeof(float));
  if (cudaMemset(hough_data, 0, N * height * width * 3 * sizeof(float)) != cudaSuccess)
    fprintf(stderr, "reset error\n");

  output_size = N * height * width;
  compute_hough_kernel<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, stream>>>(
      output_size, hough_space, hough_data, vertmap, extents, meta_data,
      arrays, array_sizes, labels, height, width, inlierThreshold, skip_pixels);
  cudaThreadSynchronize();

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed compute hough space: %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  // step 3: find the maximum in hough space
  std::vector<int> max_indexes_host(N);
  for (int i = 0; i < N; i++)
  {
    float *hmax = thrust::max_element(thrust::device, hough_space + i * height * width, hough_space + (i+1) * height * width);
    max_indexes_host[i] = hmax - hough_space;
    // printf("Max indexes %d) %d\n", i, max_indexes_host[i]);
  }

  int* max_indexes; 
  cudaMalloc((void **)&max_indexes, N * sizeof(int));
  cudaMemcpy(max_indexes, &max_indexes_host[0], N * sizeof(int), cudaMemcpyHostToDevice);

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed compute maximum: %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  // step 4: compute outputs
  output_size = N;
  compute_rois_kernel<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, stream>>>(
      output_size, top_box, top_pose, 
      poses, meta_data, hough_space, hough_data, max_indexes, labels,
      height, width);
  cudaThreadSynchronize();

  
  // err checking
  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed compute outputs: %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  cudaFree(arrays);
  cudaFree(array_sizes);
  cudaFree(hough_space);
  cudaFree(hough_data);
  cudaFree(max_indexes);

  return 1;
}


std::vector<at::Tensor> hough_voting_forward_cuda
(
    const at::Tensor& labels, const at::Tensor& masks, const at::Tensor& vertmap, const at::Tensor& extents, 
    const at::Tensor& meta_data, const at::Tensor& poses,
    const float inlierThreshold, const int skip_pixels
) 
{

  int batch_size = masks.size(0);
  int N = batch_size;
  int H = masks.size(1);
  int W = masks.size(2);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // float tensors
  at::Tensor top_box = at::zeros({N, 7}, vertmap.options());
  at::Tensor top_pose = at::zeros({N, 7}, vertmap.options());

  HoughVotingForwardLaucher(
        labels.contiguous().data<int>(), masks.contiguous().data<int>(), vertmap.contiguous().data<float>(), extents.contiguous().data<float>(), 
        meta_data.contiguous().data<float>(), poses.contiguous().data<float>(),
        N, H, W, 
        inlierThreshold, // votingThreshold, perThreshold, 
        skip_pixels,
        top_box.data<float>(), top_pose.data<float>(), 
        stream
       );
  THCudaCheck(cudaGetLastError());    

  return {top_box, top_pose};
}
