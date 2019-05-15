#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <cuda_runtime.h>
#include <cstdio>
#include <memory>

#include "rotate_rect_ops.h"

// TODO make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)


#define CUDA_CHECK(call) { \
  cudaError_t err; \
  if ((err = (call)) != cudaSuccess) { \
    fprintf(stderr, "Got error %s at %s:%d\n", cudaGetErrorString(err), \
        __FILE__, __LINE__); \
    exit(1); \
  } \
}

template <typename T>
struct device_ptr_deleter {
  void operator()(T* ptr) {
    CUDA_CHECK(cudaFree(ptr));
  }
};

template <typename T>
struct host_ptr_deleter {
  void operator()(T* ptr) {
    CUDA_CHECK(cudaFreeHost(ptr));
  }
};

template <typename T>
using unique_ptr_device = std::unique_ptr<T[], device_ptr_deleter<T>>;

template <typename T>
using unique_ptr_host = std::unique_ptr<T[], host_ptr_deleter<T>>;



const int TILE_DIM = 32;

template <typename T>
__global__ void matrix_transpose(
    T* __restrict__ transposed_matrix,
    const T* __restrict__ original_matrix,
    const int num_columns,
    const int num_rows
    )
{
  __shared__ T tile[TILE_DIM][TILE_DIM+1];

  int row = blockIdx.y * TILE_DIM + threadIdx.y;
  int column = blockIdx.x * TILE_DIM + threadIdx.x;

  if (row < num_rows && column < num_columns) {
    tile[threadIdx.y][threadIdx.x] = original_matrix[row * num_columns + column];
  }
  __syncthreads();

  int transpose_column = blockIdx.y * TILE_DIM + threadIdx.x;
  int transpose_row = blockIdx.x * TILE_DIM + threadIdx.y;
  if (transpose_row < num_columns && transpose_column < num_rows) {
    transposed_matrix[transpose_row * num_rows + transpose_column] = tile[threadIdx.x][threadIdx.y];
  }
}


template <typename T>
__device__ T bilinear_interpolate(const T* bottom_data,
    const int height, const int width,
    T y, T x,
    const int index /* index for debug only*/) {

  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    //empty
    return 0;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int y_low = (int) y;
  int x_low = (int) x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T) y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T) x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  T v1 = bottom_data[y_low * width + x_low];
  T v2 = bottom_data[y_low * width + x_high];
  T v3 = bottom_data[y_high * width + x_low];
  T v4 = bottom_data[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}



template <typename T>
__device__ void bilinear_interpolate_gradient(
    const int height, const int width,
    T y, T x,
    T & w1, T & w2, T & w3, T & w4,
    int & x_low, int & x_high, int & y_low, int & y_high,
    const int index /* index for debug only*/) {

  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    //empty
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  y_low = (int) y;
  x_low = (int) x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T) y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T) x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  // reference in forward
  // T v1 = bottom_data[y_low * width + x_low];
  // T v2 = bottom_data[y_low * width + x_high];
  // T v3 = bottom_data[y_high * width + x_low];
  // T v4 = bottom_data[y_high * width + x_high];
  // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}


template <typename T>
__global__ void RRoIAlignForward(const int nthreads, const T* bottom_data,
    const T spatial_scale, const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int sampling_ratio,
    const T* bottom_rois, T* top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;


    const T* offset_bottom_rois = bottom_rois + n * 6;  // batch_ind, xc, yc, w, h, angle
    int roi_batch_ind = offset_bottom_rois[0];

    // Force malformed ROIs to be 1x1
    T roi_width = max(offset_bottom_rois[3] * spatial_scale, (T)1.);
    T roi_height = max(offset_bottom_rois[4] * spatial_scale, (T)1.);

    const T* offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    const T mw = 1.0 / roi_bin_grid_w;
    const T mh = 1.0 / roi_bin_grid_h;

    // compute pool points
    T P[8];
    compute_roi_pool_pts(offset_bottom_rois, P, spatial_scale, pooled_height, pooled_width, ph, pw);

    // compute line params
    T line_params[4];
    for (int i = 0; i < 2; ++i)
    {
        line_params[i * 2] = P[((i + 1) * 2) % 8] - P[i * 2];
        line_params[i * 2 + 1] = P[((i + 1) * 2) % 8 + 1] - P[i * 2 + 1];
    }

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    T output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy ++) // e.g., iy = 0, 1
    {
      for (int ix = 0; ix < roi_bin_grid_w; ix ++)
      {
        const T x = P[0] + static_cast<T>(iy + 0.5) * line_params[0] * mh + static_cast<T>(ix + 0.5) * line_params[2] * mw;
        const T y = P[1] + static_cast<T>(iy + 0.5) * line_params[1] * mh + static_cast<T>(ix + 0.5) * line_params[3] * mw;

        T val = bilinear_interpolate(offset_bottom_data, height, width, y, x, index);
        output_val += val;
//        printf("%.2f\n", val);
      }
    }

    output_val /= count;

    top_data[index] = output_val;
  }
}


template <typename T>
__global__ void RRoIAlignBackwardFeature(const int nthreads, const T* top_diff,
    const int num_rois, const T spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int sampling_ratio,
    T* bottom_diff,
    const T* bottom_rois) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * 6;
    int roi_batch_ind = offset_bottom_rois[0];

    // Force malformed ROIs to be 1x1
    T roi_width = max(offset_bottom_rois[3] * spatial_scale, (T)1.);
    T roi_height = max(offset_bottom_rois[4] * spatial_scale, (T)1.);

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    const T mw = 1.0 / roi_bin_grid_w;
    const T mh = 1.0 / roi_bin_grid_h;

    T* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width;

    int top_offset    = (n * channels + c) * pooled_height * pooled_width;
    const T* offset_top_diff = top_diff + top_offset;
    const T top_diff_this_bin = offset_top_diff[ph * pooled_width + pw];

    // compute pool points
    T P[8];
    compute_roi_pool_pts(offset_bottom_rois, P, spatial_scale, pooled_height, pooled_width, ph, pw);

    // compute line params
    T line_params[4];
    for (int i = 0; i < 2; ++i)
    {
        line_params[i * 2] = P[((i + 1) * 2) % 8] - P[i * 2];
        line_params[i * 2 + 1] = P[((i + 1) * 2) % 8 + 1] - P[i * 2 + 1];
    }

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    for (int iy = 0; iy < roi_bin_grid_h; iy ++) // e.g., iy = 0, 1
    {
      for (int ix = 0; ix < roi_bin_grid_w; ix ++)
      {
        const T x = P[0] + static_cast<T>(iy + 0.5) * line_params[0] * mh + static_cast<T>(ix + 0.5) * line_params[2] * mw;
        const T y = P[1] + static_cast<T>(iy + 0.5) * line_params[1] * mh + static_cast<T>(ix + 0.5) * line_params[3] * mw;

        T w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;

        bilinear_interpolate_gradient(height, width, y, x,
            w1, w2, w3, w4,
            x_low, x_high, y_low, y_high,
            index);

        T g1 = top_diff_this_bin * w1 / count;
        T g2 = top_diff_this_bin * w2 / count;
        T g3 = top_diff_this_bin * w3 / count;
        T g4 = top_diff_this_bin * w4 / count;

        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0)
        {
          atomicAdd(offset_bottom_diff + y_low * width + x_low, static_cast<T>(g1));
          atomicAdd(offset_bottom_diff + y_low * width + x_high, static_cast<T>(g2));
          atomicAdd(offset_bottom_diff + y_high * width + x_low, static_cast<T>(g3));
          atomicAdd(offset_bottom_diff + y_high * width + x_high, static_cast<T>(g4));
        } // if
      } // ix
    } // iy
  } // CUDA_1D_KERNEL_LOOP
} // RoIAlignBackward




at::Tensor RROIAlign_forward_cuda(const at::Tensor& input,
                                const at::Tensor& rois,
                                const float spatial_scale,
                                const int pooled_height,
                                const int pooled_width,
                                const int sampling_ratio)
{
  AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(rois.type().is_cuda(), "rois must be a CUDA tensor");

  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);

  auto output = at::empty({num_rois, channels, pooled_height, pooled_width}, input.options());
  auto output_size = num_rois * pooled_height * pooled_width * channels;
//  auto argmax = at::zeros({num_rois, channels, pooled_height, pooled_width}, input.options().dtype(at::kInt));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(THCCeilDiv((long)output_size, 512L), 4096L));
  dim3 block(512);

  if (output.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES(input.type(), "RROIAlign_forward", [&] {
    RRoIAlignForward<scalar_t><<<grid, block, 0, stream>>>(
         output_size,
         input.contiguous().data<scalar_t>(),
         spatial_scale,
         channels,
         height,
         width,
         pooled_height,
         pooled_width,
         sampling_ratio,
         rois.contiguous().data<scalar_t>(),
         output.data<scalar_t>()
     );
  });
  THCudaCheck(cudaGetLastError());
  return output;
}

template <typename T>
__device__ void compute_transform_matrix(
    T* __restrict__ matrix,
    const T* __restrict__ rois,
    const float spatial_scale,
    const int pooled_height, const int pooled_width)
{
  T cx = rois[1] * spatial_scale;
  T cy = rois[2] * spatial_scale;
  // Force malformed ROIs to be 1x1
  T w = max(rois[3] * spatial_scale, T(1));
  T h = max(rois[4] * spatial_scale, T(1));
  T angle = deg2rad(rois[5]);

  // TransformPrepare
  T dx = -pooled_width / 2.0;
  T dy = -pooled_height / 2.0;
  T Sx = w / pooled_width;
  T Sy = h / pooled_height;
  T Alpha = cos(angle);
  T Beta = -sin(angle);
  T Dx = cx;
  T Dy = cy;

  matrix[0] = Alpha*Sx;
  matrix[1] = Beta*Sy;
  matrix[2] = Alpha*Sx*dx+Beta*Sy*dy+Dx;
  matrix[3] = -Beta*Sx;
  matrix[4] = Alpha*Sy;
  matrix[5] = -Beta*Sx*dx+Alpha*Sy*dy+Dy;
}


// local memory version
template <typename T>
__global__ void compute_roi_pool_pts_local(
    T* __restrict__ roi_pool_pts,
    const T* __restrict__ rois,
    const float spatial_scale,
    const int roi_pool_pt_num,
    const int num_rois,
    const int pooled_height, const int pooled_width)
{
  T matrix[6];

  // int idx = blockIdx.x * (blockDim.x * blockDim.y) + threadIdx.x * blockDim.y + threadIdx.y;
  int idx = blockIdx.x * (pooled_height * pooled_width) + threadIdx.x * pooled_width + threadIdx.y;
  if (idx >= num_rois * pooled_height * pooled_width) {
    return;
  }

  int pw = threadIdx.y;
  int ph = threadIdx.x;
  int n = blockIdx.x;

  compute_transform_matrix(
      matrix,
      rois + n*6,
      spatial_scale,
      pooled_height,
      pooled_width);

  // ORDER IN CLOCKWISE OR ANTI-CLOCKWISE
  // (0,1),(0,0),(1,0),(1,1)
  roi_pool_pts[roi_pool_pt_num * 0 + idx] = matrix[0]*pw     + matrix[1]*(ph+1) + matrix[2];
  roi_pool_pts[roi_pool_pt_num * 1 + idx] = matrix[3]*pw     + matrix[4]*(ph+1) + matrix[5];
  roi_pool_pts[roi_pool_pt_num * 2 + idx] = matrix[0]*pw     + matrix[1]*ph     + matrix[2];
  roi_pool_pts[roi_pool_pt_num * 3 + idx] = matrix[3]*pw     + matrix[4]*ph     + matrix[5];
  roi_pool_pts[roi_pool_pt_num * 4 + idx] = matrix[0]*(pw+1) + matrix[1]*ph     + matrix[2];
  roi_pool_pts[roi_pool_pt_num * 5 + idx] = matrix[3]*(pw+1) + matrix[4]*ph     + matrix[5];
  roi_pool_pts[roi_pool_pt_num * 6 + idx] = matrix[0]*(pw+1) + matrix[1]*(ph+1) + matrix[2];
  roi_pool_pts[roi_pool_pt_num * 7 + idx] = matrix[3]*(pw+1) + matrix[4]*(ph+1) + matrix[5];
}

template <typename T>
__global__ void bp_rroi_align_backward_kernel(
    T* __restrict__ bottom_diff,
    const T* __restrict__ top_diff,
    const T* __restrict__ roi_pool_pts,
    const T* __restrict__ rois,
    const float spatial_scale,
    const int sampling_ratio,
    const int num_rois,
    const int batch_size, const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width)
{
  __shared__ T roi_pool_pts_shared[8];
  __shared__ T line_params[4];
  __shared__ T rois_shared[6];

  const int roi_pool_pt_num = num_rois * pooled_height * pooled_width;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < roi_pool_pt_num; i += blockDim.x * gridDim.x) {
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (c < channels) {
      int pw = i % pooled_width;
      int ph = (i / pooled_width) % pooled_height;
      int n = i / pooled_width / pooled_height;

      const T* rois_offset = rois + n * 6;  // batch_ind, xc, yc, w, h, angle
      if (threadIdx.y < 6) {
        rois_shared[threadIdx.y] = rois_offset[threadIdx.y];
      }

      int roi_pool_idx = n * pooled_height * pooled_width + ph * pooled_width + pw;
      int roi_pool_idx_shared = threadIdx.y;
      if (roi_pool_idx_shared < 8) {
        roi_pool_pts_shared[roi_pool_idx_shared] = roi_pool_pts[roi_pool_idx_shared * roi_pool_pt_num + roi_pool_idx];
      }
      __syncthreads();

      // compute line params
      // if (roi_pool_idx_shared < 4) {
      //   line_params[roi_pool_idx_shared] = roi_pool_pts_shared[((roi_pool_idx_shared / 2) + 1) * 2 % 8 + roi_pool_idx_shared % 2] - roi_pool_pts_shared[roi_pool_idx_shared];
      // }
      if (roi_pool_idx_shared < 2) {
        line_params[roi_pool_idx_shared * 2] = roi_pool_pts_shared[((roi_pool_idx_shared + 1) * 2) % 8] - roi_pool_pts_shared[roi_pool_idx_shared * 2];
        line_params[roi_pool_idx_shared * 2 + 1] = roi_pool_pts_shared[((roi_pool_idx_shared + 1) * 2) % 8 + 1] - roi_pool_pts_shared[roi_pool_idx_shared * 2 + 1];
      }
      __syncthreads();

      int roi_batch_id = rois_shared[0];

      // Force malformed ROIs to be 1x1
      T roi_width = max(rois_shared[3] * spatial_scale, (T)1.);
      T roi_height = max(rois_shared[4] * spatial_scale, (T)1.);
      // We use roi_bin_grid to sample the grid and mimic integral
      int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height); // e.g., = 2
      int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

      const T mw = 1.0 / roi_bin_grid_w;
      const T mh = 1.0 / roi_bin_grid_h;

      int top_data_idx = (n * channels + c) * pooled_width * pooled_height + ph * pooled_width + pw;
      const T top_diff_this_bin = top_diff[top_data_idx];

      // We do average (integral) pooling inside a bin
      const T count = roi_bin_grid_h * roi_bin_grid_w;

      for (int iy = 0; iy < roi_bin_grid_h; iy ++)
      {
        for (int ix = 0; ix < roi_bin_grid_w; ix ++)
        {
          const T x = roi_pool_pts_shared[0] + static_cast<T>(iy + 0.5) * line_params[0] * mh + static_cast<T>(ix + 0.5) * line_params[2] * mw;
          const T y = roi_pool_pts_shared[1] + static_cast<T>(iy + 0.5) * line_params[1] * mh + static_cast<T>(ix + 0.5) * line_params[3] * mw;

          T w1, w2, w3, w4;
          int x_low, x_high, y_low, y_high;

          bilinear_interpolate_gradient(height, width, y, x,
              w1, w2, w3, w4,
              x_low, x_high, y_low, y_high,
              0);

          T g1 = top_diff_this_bin * w1 / count;
          T g2 = top_diff_this_bin * w2 / count;
          T g3 = top_diff_this_bin * w3 / count;
          T g4 = top_diff_this_bin * w4 / count;

          if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
            atomicAdd(bottom_diff + ((y_low  * width + x_low ) * batch_size + roi_batch_id) * channels + c, static_cast<T>(g1));
            atomicAdd(bottom_diff + ((y_low  * width + x_high) * batch_size + roi_batch_id) * channels + c, static_cast<T>(g2));
            atomicAdd(bottom_diff + ((y_high * width + x_low ) * batch_size + roi_batch_id) * channels + c, static_cast<T>(g3));
            atomicAdd(bottom_diff + ((y_high * width + x_high) * batch_size + roi_batch_id) * channels + c, static_cast<T>(g4));
          }

        }
      }
    }
  }
}



void bp_rroi_align_backward(
    int batch_size,
    int num_rois,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    float spatial_scale,
    const float* top_diff_d,
    const float* rois_d,
    float* bottom_diff_d,
    cudaStream_t stream
    )
{
  unique_ptr_device<float> roi_pool_pts_d(nullptr);
  int roi_pool_pt_num = num_rois * pooled_height * pooled_width;
  CUDA_CHECK(cudaMalloc((void **) &roi_pool_pts_d, 8 * roi_pool_pt_num * sizeof(float)));

  unique_ptr_device<float> bottom_diff_coalesced_d(nullptr);
  auto bottom_data_size = batch_size * channels * height * width;
  CUDA_CHECK(cudaMalloc((void **) &bottom_diff_coalesced_d, bottom_data_size * sizeof(float)));

  {
    int block_x = TILE_DIM;
    int block_y = TILE_DIM;
    const int num_columns = height * width;
    const int num_rows = batch_size * channels;
    int grid_x = static_cast<int>(std::ceil(num_columns * 1.0 / block_x));
    int grid_y = static_cast<int>(std::ceil(num_rows * 1.0 / block_y));
    dim3 block(block_x, block_y);
    dim3 grid(grid_x, grid_y);
    matrix_transpose<float><<<grid, block, 0, stream>>>(
        bottom_diff_coalesced_d.get(),
        bottom_diff_d,
        num_columns,
        num_rows
        );
  }

  {
    dim3 block(pooled_height, pooled_width);
    dim3 grid(num_rois);
    compute_roi_pool_pts_local<float><<<grid, block, 0, stream>>>(
        roi_pool_pts_d.get(),
        rois_d,
        spatial_scale,
        roi_pool_pt_num,
        num_rois,
        pooled_height,
        pooled_width);
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  {
    // cudaDeviceProp deviceProperties;
    // int gpu_id = 0;
    // CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, gpu_id));

    int max_thread_num = 256;
    // int thread_num_x = std::min(max_thread_num / 8, pooled_width);
    // int thread_num_y = std::min(max_thread_num / thread_num_x, channels);
    int thread_num_y = std::min(channels, max_thread_num);
    // int thread_num_x = max_thread_num / thread_num_y;
    int thread_num_x = 1;
    // int block_num_x = std::min(static_cast<int>(std::ceil(pooled_width * pooled_height * num_rois * 1.0 / thread_num_x)), deviceProperties.maxGridSize[0]);
    int block_num_x = std::min(static_cast<int>(std::ceil(pooled_width * pooled_height * num_rois * 1.0 / thread_num_x)), 65535);
    int block_num_y = static_cast<int>(std::ceil(channels * 1.0 / thread_num_y));
    dim3 block(thread_num_x, thread_num_y);
    dim3 grid(block_num_x, block_num_y);
    int sampling_ratio = 0; // default
    bp_rroi_align_backward_kernel<float><<<grid, block, 0, stream>>>(
        bottom_diff_coalesced_d.get(),
        top_diff_d,
        roi_pool_pts_d.get(),
        rois_d,
        spatial_scale,
        sampling_ratio,
        num_rois,
        batch_size,
        channels,
        height,
        width,
        pooled_height,
        pooled_width);
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  {
    int block_x = TILE_DIM;
    int block_y = TILE_DIM;
    const int num_columns = batch_size * channels;
    const int num_rows = height * width;
    int grid_x = static_cast<int>(std::ceil(num_columns * 1.0 / block_x));
    int grid_y = static_cast<int>(std::ceil(num_rows * 1.0 / block_y));
    dim3 block(block_x, block_y);
    dim3 grid(grid_x, grid_y);
    matrix_transpose<float><<<grid, block, 0, stream>>>(
        bottom_diff_d,
        bottom_diff_coalesced_d.get(),
        num_columns,
        num_rows
        );
  }
}


at::Tensor RROIAlign_backward_cuda(const at::Tensor& grad,
                      const at::Tensor& rois,
                      const float spatial_scale,
                      const int pooled_height,
                      const int pooled_width,
                      const int batch_size,
                      const int channels,
                      const int height,
                      const int width,
                      const int sampling_ratio)
{
  AT_ASSERTM(grad.type().is_cuda(), "grad must be a CUDA tensor");
  AT_ASSERTM(rois.type().is_cuda(), "rois must be a CUDA tensor");

  auto num_rois = rois.size(0);
  auto grad_input = at::zeros({batch_size, channels, height, width}, grad.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(THCCeilDiv((long)grad.numel(), 512L), 4096L));
  dim3 block(512);

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return grad_input;
  }

  AT_DISPATCH_FLOATING_TYPES(grad.type(), "RROIAlign_backward", [&] {
//    bp_rroi_align_backward(
//      batch_size,
//      num_rois,
//      channels,
//      height,
//      width,
//      pooled_height,
//      pooled_width,
//      spatial_scale,
//      grad.contiguous().data<float>(),
//      rois.contiguous().data<float>(),
//      grad_input.data<float>(),
//      stream);
     RRoIAlignBackwardFeature<scalar_t><<<grid, block, 0, stream>>>(
         grad.numel(),
         grad.contiguous().data<scalar_t>(),
         num_rois,
         spatial_scale,
         channels,
         height,
         width,
         pooled_height,
         pooled_width,
         sampling_ratio,
         grad_input.data<scalar_t>(),
         rois.contiguous().data<scalar_t>());
  });
  THCudaCheck(cudaGetLastError());
  return grad_input;
}
