// #ifdef __cplusplus
// extern "C" {
// #endif
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>


#include <stdio.h>
#include <vector>
#include <math.h>
#include <float.h>
#include <cassert>

#include <thrust/device_vector.h>
// #include <thrust/copy.h>
#include <thrust/extrema.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 512
#define POSE_CHANNELS 4

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

 // CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

inline
cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", 
            cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}


template <typename Dtype>
__global__ void AveragedistanceBackward(const int nthreads, const Dtype* top_diff,
    const Dtype* bottom_diff, Dtype* output) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    output[index] = top_diff[0] * bottom_diff[index];
  }
}

 
int AveragedistanceBackwardLaucher(const float* top_diff, const float* bottom_diff, const int batch_size,
    const int channels, float* output, cudaStream_t stream)
{
  const int kThreadsPerBlock = THREADS_PER_BLOCK;
  const int output_size = batch_size * channels;
  cudaError_t err;

  AveragedistanceBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, stream>>>(
      output_size, top_diff, bottom_diff, output);

  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  return 1;
}


at::Tensor ave_dist_loss_backward_cuda(const at::Tensor& grad, const at::Tensor& bottom_diff) 
{
  at::Tensor output = at::zeros_like(bottom_diff);

  int batch_size = bottom_diff.size(0);
  int channels = bottom_diff.size(1);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AveragedistanceBackwardLaucher(
    grad.contiguous().data<float>(), bottom_diff.contiguous().data<float>(), 
    batch_size, channels, output.data<float>(), stream
  );
  THCudaCheck(cudaGetLastError());

  return output;
}

// NEW
template <typename Dtype>
__global__ void sum_losses_gradients(const int nthreads, const Dtype* losses, const Dtype* diffs, const int batch_size, 
    const int num_points, Dtype* loss_batch, Dtype* bottom_diff) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    int n = index / POSE_CHANNELS;
    int c = index % POSE_CHANNELS;

    bottom_diff[index] = 0;
    for (int p = 0; p < num_points; p++)
    {
      int index_diff = n * num_points * POSE_CHANNELS + p * POSE_CHANNELS + c;
      bottom_diff[index] += diffs[index_diff];
    }

    if (c == 0)
    {
      loss_batch[n] = 0;
      for (int p = 0; p < num_points; p++)
        loss_batch[n] += losses[n * num_points + p];
    }

  }
}

template <typename Dtype>
__global__ void AveragedistanceForward(const int nthreads, const Dtype* prediction, const Dtype* target,
    const int* labels, const Dtype* point, const Dtype* symmetry, const int batch_size, 
    const int num_points, const float margin, Dtype* rotations, Dtype* losses, Dtype* diffs) 
{
  CUDA_1D_KERNEL_LOOP(index_thread, nthreads) 
  {
    // batch index
    int n = index_thread / num_points;

    int index_cls = labels[n];

    if (index_cls <= 0)  // 0 for bg class, TODO: REMOVE?
      return;

    // point index
    int p = index_thread % num_points;

    Dtype s, u, v, w;

    int index = n * POSE_CHANNELS;
    // gt quaternion
    s = target[index+0];
    u = target[index+1];
    v = target[index+2];
    w = target[index+3];

    // gt rotation matrix
    int ind = n * num_points * 6 * 9 + p * 6 * 9;
    rotations[ind + 0] = s * s + u * u - v * v - w * w;
    rotations[ind + 1] = 2 * (u * v - s * w);
    rotations[ind + 2] = 2 * (u * w + s * v);
    rotations[ind + 3] = 2 * (u * v + s * w);
    rotations[ind + 4] = s * s - u * u + v * v - w * w;
    rotations[ind + 5] = 2 * (v * w - s * u);
    rotations[ind + 6] = 2 * (u * w - s * v);
    rotations[ind + 7] = 2 * (v * w + s * u);
    rotations[ind + 8] = s * s - u * u - v * v + w * w;

    // predicted quaternion
    s = prediction[index + 0];
    u = prediction[index + 1];
    v = prediction[index + 2];
    w = prediction[index + 3];

    // predicted rotation matrix
    ind = n * num_points * 6 * 9 + p * 6 * 9 + 9;
    rotations[ind + 0] = s * s + u * u - v * v - w * w;
    rotations[ind + 1] = 2 * (u * v - s * w);
    rotations[ind + 2] = 2 * (u * w + s * v);
    rotations[ind + 3] = 2 * (u * v + s * w);
    rotations[ind + 4] = s * s - u * u + v * v - w * w;
    rotations[ind + 5] = 2 * (v * w - s * u);
    rotations[ind + 6] = 2 * (u * w - s * v);
    rotations[ind + 7] = 2 * (v * w + s * u);
    rotations[ind + 8] = s * s - u * u - v * v + w * w;

    // derivatives of Ru to quaternion
    ind = n * num_points * 6 * 9 + p * 6 * 9 + 18;
    rotations[ind + 0] = 2 * s;
    rotations[ind + 1] = -2 * w;
    rotations[ind + 2] = 2 * v;
    rotations[ind + 3] = 2 * w;
    rotations[ind + 4] = 2 * s;
    rotations[ind + 5] = -2 * u;
    rotations[ind + 6] = -2 * v;
    rotations[ind + 7] = 2 * u;
    rotations[ind + 8] = 2 * s;

    ind = n * num_points * 6 * 9 + p * 6 * 9 + 27;
    rotations[ind + 0] = 2 * u;
    rotations[ind + 1] = 2 * v;
    rotations[ind + 2] = 2 * w;
    rotations[ind + 3] = 2 * v;
    rotations[ind + 4] = -2 * u;
    rotations[ind + 5] = -2 * s;
    rotations[ind + 6] = 2 * w;
    rotations[ind + 7] = 2 * s;
    rotations[ind + 8] = -2 * u;

    ind = n * num_points * 6 * 9 + p * 6 * 9 + 36;
    rotations[ind + 0] = -2 * v;
    rotations[ind + 1] = 2 * u;
    rotations[ind + 2] = 2 * s;
    rotations[ind + 3] = 2 * u;
    rotations[ind + 4] = 2 * v;
    rotations[ind + 5] = 2 * w;
    rotations[ind + 6] = -2 * s;
    rotations[ind + 7] = 2 * w;
    rotations[ind + 8] = -2 * v;

    ind = n * num_points * 6 * 9 + p * 6 * 9 + 45;
    rotations[ind + 0] = -2 * w;
    rotations[ind + 1] = -2 * s;
    rotations[ind + 2] = 2 * u;
    rotations[ind + 3] = 2 * s;
    rotations[ind + 4] = -2 * w;
    rotations[ind + 5] = 2 * v;
    rotations[ind + 6] = 2 * u;
    rotations[ind + 7] = 2 * v;
    rotations[ind + 8] = 2 * w;

    // for the point
    index = index_cls * num_points * 3 + p * 3;
    ind = n * num_points * 6 * 9 + p * 6 * 9;

    // rotate the first point
    Dtype x1 = rotations[ind + 9 + 0] * point[index + 0] + rotations[ind + 9 + 1] * point[index + 1] + rotations[ind + 9 + 2] * point[index + 2];
    Dtype y1 = rotations[ind + 9 + 3] * point[index + 0] + rotations[ind + 9 + 4] * point[index + 1] + rotations[ind + 9 + 5] * point[index + 2];
    Dtype z1 = rotations[ind + 9 + 6] * point[index + 0] + rotations[ind + 9 + 7] * point[index + 1] + rotations[ind + 9 + 8] * point[index + 2];

    int index_min;
    Dtype x2, y2, z2;
    if (symmetry[index_cls] > 0)
    {
      // find the closet point for symmetry object
      Dtype dmin = FLT_MAX;
      for (int i = 0; i < num_points; i++)
      {
        int index2 = index_cls * num_points * 3 + i * 3;
        x2 = rotations[ind + 0] * point[index2 + 0] + rotations[ind + 1] * point[index2 + 1] + rotations[ind + 2] * point[index2 + 2];
        y2 = rotations[ind + 3] * point[index2 + 0] + rotations[ind + 4] * point[index2 + 1] + rotations[ind + 5] * point[index2 + 2];
        z2 = rotations[ind + 6] * point[index2 + 0] + rotations[ind + 7] * point[index2 + 1] + rotations[ind + 8] * point[index2 + 2];
        Dtype distance = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2);
        if (distance < dmin)
        {
          dmin = distance;
          index_min = index2;
        }
      }
    }
    else
      index_min = index;

    x2 = rotations[ind + 0] * point[index_min + 0] + rotations[ind + 1] * point[index_min + 1] + rotations[ind + 2] * point[index_min + 2];
    y2 = rotations[ind + 3] * point[index_min + 0] + rotations[ind + 4] * point[index_min + 1] + rotations[ind + 5] * point[index_min + 2];
    z2 = rotations[ind + 6] * point[index_min + 0] + rotations[ind + 7] * point[index_min + 1] + rotations[ind + 8] * point[index_min + 2];

    Dtype distance = ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
    if (distance < margin)
      continue;

    losses[index_thread] = (distance - margin) / (2.0 * batch_size * num_points);


    int index_diff = n * num_points * POSE_CHANNELS + p * POSE_CHANNELS;
    for (int j = 0; j < 3; j++)
    {
      Dtype diff;
      if (j == 0)
        diff = x1 - x2;
      else if (j == 1)
        diff = y1 - y2;
      else
        diff = z1 - z2;
      for (int k = 0; k < 3; k++)
      {
        ind = n * num_points * 6 * 9 + p * 6 * 9 + 18;
        diffs[index_diff + 0] += diff * point[index + k] * rotations[ind + j * 3 + k] / (batch_size * num_points);
        ind = n * num_points * 6 * 9 + p * 6 * 9 + 27;
        diffs[index_diff + 1] += diff * point[index + k] * rotations[ind + j * 3 + k] / (batch_size * num_points);
        ind = n * num_points * 6 * 9 + p * 6 * 9 + 36;
        diffs[index_diff + 2] += diff * point[index + k] * rotations[ind + j * 3 + k] / (batch_size * num_points);
        ind = n * num_points * 6 * 9 + p * 6 * 9 + 45;
        diffs[index_diff + 3] += diff * point[index + k] * rotations[ind + j * 3 + k] / (batch_size * num_points);
      }
    }

  }
}

std::vector<at::Tensor> ave_dist_loss_forward_cuda(
    const at::Tensor& poses_pred, const at::Tensor& poses_target, const at::Tensor& poses_labels, const at::Tensor& points, const at::Tensor& symmetry,
    const float margin)
{

  int batch_size = poses_pred.size(0);
  int num_points = points.size(1);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // float tensors
  at::Tensor loss_tensor = at::zeros({1}, poses_pred.options());
  at::Tensor bottom_diff_tensor = at::zeros_like(poses_pred);

  float* top_data = loss_tensor.data<float>();
  float* bottom_diff = bottom_diff_tensor.data<float>();

  const float* bottom_prediction = poses_pred.contiguous().data<float>(); 
  const float* bottom_target = poses_target.contiguous().data<float>(); 
  const float* bottom_point = points.contiguous().data<float>(); 
  const float* bottom_symmetry = symmetry.contiguous().data<float>(); 

  const int* bottom_labels = poses_labels.contiguous().data<int>(); 

 // run kernels
  cudaError_t err;
  const int kThreadsPerBlock = THREADS_PER_BLOCK;

  /*
  temp losses
  */
  float* losses;
  checkCuda(cudaMalloc((void **)&losses, batch_size * num_points * sizeof(float)));
  checkCuda(cudaMemset(losses, 0, batch_size * num_points * sizeof(float)));

  float* loss_batch;
  checkCuda(cudaMalloc((void **)&loss_batch, batch_size * sizeof(float)));
  checkCuda(cudaMemset(loss_batch, 0, batch_size * sizeof(float)));

  /*
  temp diffs
  */
  int output_size = batch_size * num_points * POSE_CHANNELS;
  float *diffs;
  checkCuda(cudaMalloc((void **)&diffs, output_size * sizeof(float)));
  checkCuda(cudaMemset(diffs, 0, output_size * sizeof(float)));  

  /*
  temp rotations
  */
  output_size = batch_size * num_points * 6 * 9;
  float* rotations;
  checkCuda(cudaMalloc((void **)&rotations, output_size * sizeof(float)));
  checkCuda(cudaMemset(rotations, 0, output_size * sizeof(float)));  

  /*
  compute the losses and gradients
  */
  output_size = batch_size * num_points;
  AveragedistanceForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, stream>>>(
      output_size, bottom_prediction, bottom_target, bottom_labels, bottom_point, bottom_symmetry,
      batch_size, num_points, margin, rotations, losses, diffs);
  cudaDeviceSynchronize();

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed: %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  /*
  sum the diffs
  */
  output_size = batch_size * POSE_CHANNELS;
  sum_losses_gradients<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, stream>>>(
      output_size, losses, diffs, batch_size, num_points, loss_batch, bottom_diff);
  cudaDeviceSynchronize();

  /*
  sum the loss
  */
  checkCuda(cudaMemset(top_data, 0, sizeof(float)));
  thrust::device_ptr<float> losses_ptr(loss_batch);
  float loss_host = thrust::reduce(losses_ptr, losses_ptr + batch_size);
  checkCuda(cudaMemcpy(top_data, &loss_host, sizeof(float), cudaMemcpyHostToDevice));

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed: %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  checkCuda(cudaFree(losses));
  checkCuda(cudaFree(loss_batch));
  checkCuda(cudaFree(diffs));
  checkCuda(cudaFree(rotations));

  return {loss_tensor, bottom_diff_tensor};
}

