/**
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

namespace {

#define CUDA_1D_KERNEL_LOOP(i, n)                                 \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
    i += blockDim.x * gridDim.x)

#define CUDA_2D_KERNEL_LOOP(i, n, j, m)                           \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
    i += blockDim.x * gridDim.x)         \
  for (size_t j = blockIdx.y * blockDim.y + threadIdx.y; j < (m); \
    j += blockDim.y * gridDim.y)

// The number of cuda threads to use. Since work is assigned to SMs at the
// granularity of a block, 128 is chosen to allow utilizing more SMs for
// smaller input sizes.
// 1D grid
constexpr int CAFFE_CUDA_NUM_THREADS = 128;
// 2D grid
constexpr int CAFFE_CUDA_NUM_THREADS_2D_DIMX = 16;
constexpr int CAFFE_CUDA_NUM_THREADS_2D_DIMY = 16;
const dim3 CAFFE_CUDA_NUM_THREADS_2D = {CAFFE_CUDA_NUM_THREADS_2D_DIMX,CAFFE_CUDA_NUM_THREADS_2D_DIMY,1};

constexpr int CAFFE_MAXIMUM_NUM_BLOCKS = 4096;
// 2D grid
constexpr int CAFFE_MAXIMUM_NUM_BLOCKS_2D_DIMX = 128;
constexpr int CAFFE_MAXIMUM_NUM_BLOCKS_2D_DIMY = 128;
const dim3 CAFFE_MAXIMUM_NUM_BLOCKS_2D = {CAFFE_MAXIMUM_NUM_BLOCKS_2D_DIMX,CAFFE_MAXIMUM_NUM_BLOCKS_2D_DIMX,1};

inline int CAFFE_GET_BLOCKS(const int N) {
  return std::max(
      std::min(
        (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS,
        CAFFE_MAXIMUM_NUM_BLOCKS),
      // Use at least 1 block, since CUDA does not allow empty block
      1);
}

/**
  * @brief Compute the number of blocks needed to run N threads for a 2D grid
  */
inline dim3 CAFFE_GET_BLOCKS_2D(const int N, const int M) {
  dim3 grid;
  // Not calling the 1D version for each dim to keep all constants as literals

  grid.x = std::max(
      std::min(
        (N + CAFFE_CUDA_NUM_THREADS_2D_DIMX - 1) / CAFFE_CUDA_NUM_THREADS_2D_DIMX,
        CAFFE_MAXIMUM_NUM_BLOCKS_2D_DIMX),
      // Use at least 1 block, since CUDA does not allow empty block
      1);

  grid.y = std::max(
      std::min(
        (N + CAFFE_CUDA_NUM_THREADS_2D_DIMY - 1) / CAFFE_CUDA_NUM_THREADS_2D_DIMY,
        CAFFE_MAXIMUM_NUM_BLOCKS_2D_DIMY),
      // Use at least 1 block, since CUDA does not allow empty block
      1);

  return grid;
}

/**
 * d_sorted_score_keys -- indexes into _original_ scores
 * nboxes_to_generate -- pre_nms_topn
 */
__global__ void GeneratePreNMSUprightBoxesKernel(
    const long *d_sorted_scores_keys,
		const int nboxes_to_generate,
		const float *d_bbox_deltas,   // [N, A*4, H, W]
		const float4 *d_anchors,
		const int H,
		const int W,
		const int K, // K = H*W
		const int A,
		const int KA, // KA = K*A
		const float feat_stride,
		const float min_size,
		const float *d_img_info_vec,
		const int num_images,
		const float bbox_xform_clip,
		const bool correct_transform,
		float4 *d_out_boxes,
		const int prenms_nboxes, // leading dimension of out_boxes
		float *d_inout_scores, // [N, A, H, W]
		int *d_boxes_keep_flags) {
  // Going to generate pre_nms_nboxes boxes per image
  for (int ibox = blockIdx.x * blockDim.x + threadIdx.x; ibox < nboxes_to_generate;
    ibox += blockDim.x * gridDim.x) {
    for (int image_index = blockIdx.y * blockDim.y + threadIdx.y; image_index < num_images;
        image_index += blockDim.y * gridDim.y) {
#if 0
      printf("ibox: %d, nboxes_to_generate: %d, image_index: %d, num_images: %d\n", ibox, nboxes_to_generate, image_index, num_images);
#endif
      // box_conv_index : # of the same box, but indexed in
      // the scores from the conv layer, of shape (A,H,W)
      // the num_images dimension was already removed
      // box_conv_index = a*K + h*W + w
      // Note: PyT code takes topK, so need to adjust the indexing for multi-image
      // box_conv_index is _local_ to the image_index, need to adjust into global arrays
      const int box_conv_index = d_sorted_scores_keys[image_index * prenms_nboxes + ibox];

      // We want to decompose box_conv_index in (a,h,w)
      // such as box_conv_index = a*K + h*W + w
      // (avoiding modulos in the process)
      int remaining = box_conv_index;
      const int dA = K; // stride of A
      const int a = remaining / dA;
      remaining -= a*dA;
      const int dH = W; // stride of H
      const int h = remaining / dH;
      remaining -= h*dH;
      const int w = remaining; // dW = 1

      // Order of anchors is [N, H, W, A, 4]
      const int a_idx = h * W * A + w * A + a;
      // const int a_idx = a * H * W + h * W + w;
      if (a_idx < 0 || a_idx > A * H * W) {
        printf("box_conv_index: %d, idx: %d\n", box_conv_index, image_index*KA+ibox);
        printf("F (a: %d, h: %d, w: %d, K: %d, H: %d, W: %d) -> %d\n", a, h, w, K, H, W, a_idx);
      }
      const float4 anchor = d_anchors[image_index * KA + a_idx];
      // x1,y1,x2,y2 :coordinates of anchor a, shifted for position (h,w)
      // TODO(slayton): What are these feat_stride, and where do they come from?
      const float shift_w = feat_stride * w;
      float x1 = shift_w + anchor.x;
      float x2 = shift_w + anchor.z;
      const float shift_h = feat_stride * h;
      float y1 = shift_h + anchor.y;
      float y2 = shift_h + anchor.w;

      // Deltas for that box
      // Deltas of shape (num_images,4*A,K)
      // We're going to compute 4 scattered reads
      // better than the alternative, ie transposing the complete deltas
      // array first
      int deltas_idx = image_index * (KA*4) + a*4*K+h*W+w;
      const float dx = d_bbox_deltas[deltas_idx];
      // Stride of K between each dimension
      deltas_idx += K; const float dy = d_bbox_deltas[deltas_idx];
      deltas_idx += K; float dw = d_bbox_deltas[deltas_idx];
      deltas_idx += K; float dh = d_bbox_deltas[deltas_idx];

      // Upper bound on dw,dh
      dw = fmin(dw, bbox_xform_clip);
      dh = fmin(dh, bbox_xform_clip);

      // Applying the deltas
      float width = x2 - x1 + 1.0f;
      const float ctr_x = x1 + 0.5f*width;
      const float pred_ctr_x = ctr_x + width*dx;
      const float pred_w = width*expf(dw);
      x1 = pred_ctr_x - 0.5f*pred_w;
      x2 = pred_ctr_x + 0.5f*pred_w;

      float height = y2 - y1 + 1.0f;
      const float ctr_y = y1 + 0.5f*height;
      const float pred_ctr_y = ctr_y + height*dy;
      const float pred_h = height*expf(dh);
      y1 = pred_ctr_y - 0.5f*pred_h;
      y2 = pred_ctr_y + 0.5f*pred_h;

      if(correct_transform) {
        x2 -= 1.0f;
        y2 -= 1.0f;
      }

      // End of box_coder.decode(..) part

      // Clipping box to image
      // p = _clip_box_to_image(proposal, height, width)
      const float img_height = d_img_info_vec[2*image_index+0];
      const float img_width = d_img_info_vec[2*image_index+1];
      const float min_size_scaled = min_size * 1.; // d_img_info_vec[3*image_index+2];
      x1 = fmax(fmin(x1, img_width-1.0f), 0.0f);
      y1 = fmax(fmin(y1, img_height-1.0f), 0.0f);
      x2 = fmax(fmin(x2, img_width-1.0f), 0.0f);
      y2 = fmax(fmin(y2, img_height-1.0f), 0.0f);

      // Filter boxes
      // Removing boxes with one dim < min_size
      // (center of box is in image, because of previous step)
      // keep = _filter_boxes(p, self.min_size, im_shape)
      width = x2 - x1 + 1.0f; // may have changed
      height = y2 - y1 + 1.0f;
      bool keep_box = fmin(width, height) >= min_size_scaled;
      // We are not deleting the box right now even if !keep_box
      // we want to keep the relative order of the elements stable
      // we'll do it in such a way later
      // d_boxes_keep_flags size: (num_images,prenms_nboxes)
      // d_out_boxes size: (num_images,prenms_nboxes)
      const int out_index = image_index * prenms_nboxes + ibox;
      d_boxes_keep_flags[out_index] = keep_box;
      d_out_boxes[out_index] = {x1,y1,x2,y2};

      // d_inout_scores size: (num_images,KA)
      // In PyT code this part doesn't happen
      if (!keep_box) {
        d_inout_scores[image_index * prenms_nboxes+ibox] = FLT_MIN; // for NMS
      }
    }
  }
}

__global__ void AddImageIndexToOutput(const int nboxes,
		const int image_index,
		const float4 *d_image_out_rois_without_image_index,
		float *d_image_out_rois) {
	CUDA_1D_KERNEL_LOOP(i, nboxes) {
		const float4 box = d_image_out_rois_without_image_index[i];
		// Scattered memory accesses
		// postnms_nboxes is small anyway
		const int base_idx = 5*i;
		d_image_out_rois[base_idx+0] = image_index;
		d_image_out_rois[base_idx+1] = box.x;
		d_image_out_rois[base_idx+2] = box.y;
		d_image_out_rois[base_idx+3] = box.z;
		d_image_out_rois[base_idx+4] = box.w;
	}
}

__global__ void InitializeDataKernel(const int num_images,
		const int KA,
		int *d_image_offsets,
		int *d_boxes_keys_iota,
		int *d_counters) {
	CUDA_2D_KERNEL_LOOP(box_idx, KA, img_idx, num_images) {
		d_boxes_keys_iota[img_idx*KA+box_idx] = box_idx;

		// One 1D line sets the 1D data
		if(box_idx == 0) {
			d_counters[img_idx] = 0;
			d_image_offsets[img_idx] = KA*img_idx;
			// One thread sets the last+1 offset
			if(img_idx == 0)
				d_image_offsets[num_images] = KA*num_images;
		}
	}
}

struct __align__(16) Box {
	float x1,y1,x2,y2;
};

#define BOXES_PER_THREAD (8*sizeof(int))
#define CHUNK_SIZE 2000

__launch_bounds__(CAFFE_CUDA_NUM_THREADS_2D_DIMX*CAFFE_CUDA_NUM_THREADS_2D_DIMY,4)
	__global__ void NMSKernel(const Box *d_boxes, // TODO sorted_boxes
			const int nboxes,
			const float thresh,
			const int mask_ld,
			int *d_delete_mask) {
		// Storing boxes used by this CUDA block in the shared memory
		__shared__ Box shared_i_boxes[CAFFE_CUDA_NUM_THREADS_2D_DIMX];
		// Same thing with areas
		__shared__ float shared_i_areas[CAFFE_CUDA_NUM_THREADS_2D_DIMX];
		// The condition of the for loop is common to all threads in the block
		// This is necessary to be able to call __syncthreads() inside of the loop
		for(int i_block_offset = blockIdx.x*blockDim.x;
				i_block_offset < nboxes;
				i_block_offset += blockDim.x*gridDim.x) {
			const int i_to_load = i_block_offset + threadIdx.x;
			if(i_to_load < nboxes) {
				// One 1D line load the boxes for x-dimension
				if(threadIdx.y == 0) {
					const Box box = d_boxes[i_to_load];
					shared_i_areas[threadIdx.x] = (box.x2 - box.x1 + 1.0f) * (box.y2 - box.y1 + 1.0f);
					shared_i_boxes[threadIdx.x] = box;
				}
			}
			__syncthreads();
			const int i = i_block_offset + threadIdx.x;
			for(int j_thread_offset = BOXES_PER_THREAD*(blockIdx.y*blockDim.y+threadIdx.y);
					j_thread_offset < nboxes;
					j_thread_offset += BOXES_PER_THREAD*blockDim.y*gridDim.y) {
				// Note : We can do everything using multiplication,
				// and use fp16 - we are comparing against a low precision
				// threshold
				int above_thresh = 0;
				bool valid = false;
				for(int ib=0; ib<BOXES_PER_THREAD; ++ib) {
					// This thread will compare Box i and Box j
					const int j = j_thread_offset+ib;
					if(i < j && i < nboxes && j < nboxes) {
						valid = true;
						const Box j_box = d_boxes[j];
						const Box i_box = shared_i_boxes[threadIdx.x];
						const float j_area = (j_box.x2 - j_box.x1 + 1.0f) * (j_box.y2 - j_box.y1 + 1.0f);
						const float i_area = shared_i_areas[threadIdx.x];
						// The following code will not be valid with empty boxes
						if(i_area == 0.0f || j_area == 0.0f)
							continue;
						const float xx1 = fmaxf(i_box.x1, j_box.x1);
						const float yy1 = fmaxf(i_box.y1, j_box.y1);
						const float xx2 = fminf(i_box.x2, j_box.x2);
						const float yy2 = fminf(i_box.y2, j_box.y2);

						// fdimf computes the positive difference between xx2+1 and xx1
						const float w = fdimf(xx2 + 1.0f, xx1);
						const float h = fdimf(yy2 + 1.0f, yy1);
						const float intersection = w*h;

						// Testing for a/b > t
						// eq with a > b*t (b is !=0)
						// avoiding divisions
						const float a = intersection;
						const float b = i_area + j_area - intersection;
						const float bt = b * thresh;
						// eq. to if ovr > thresh
						if(a > bt) {
							// we have score[j] <= score[i]
							above_thresh |= (1U << ib);
						}
					}
				}
				if(valid)
					d_delete_mask[i*mask_ld+j_thread_offset/BOXES_PER_THREAD] = above_thresh;
			}
			__syncthreads(); // making sure everyone is done reading smem
		}
	}

} // namespace

namespace utils {
/**
 * Perform NMS
 * In: Sorted boxes
 * In: Number of boxes (pre_nms_top_n)
 * In: IoU threshold
 * Out: Sorted indices to keep
 */
at::Tensor nms_gpu_upright(
		at::Tensor& boxes,
		const int N,
		const float thresh) {
	// Making sure we respect the __align(16)__ we promised to the compiler
	//auto iptr = reinterpret_cast<std::uintptr_t>(d_sorted_boxes_float_ptr);
	//CAFFE_ENFORCE_EQ(iptr%16, 0);

	// The next kernel expects squares
	//CAFFE_ENFORCE_EQ(CAFFE_CUDA_NUM_THREADS_2D_DIMX,CAFFE_CUDA_NUM_THREADS_2D_DIMY);

  const int mask_ld = (N+BOXES_PER_THREAD-1)/BOXES_PER_THREAD;
	const Box *d_boxes = reinterpret_cast<const Box*>(boxes.data<float>());

  // Create temporary to hold the delete mask on device
  at::Tensor dev_delete_mask = at::zeros({N * mask_ld}, at::CUDA(at::kInt));
  // similar for the host
  at::Tensor host_delete_mask = at::zeros({N * mask_ld}, at::CPU(at::kInt));

	int *d_delete_mask = dev_delete_mask.data<int>();
	int *h_delete_mask = host_delete_mask.data<int>();

  auto stream = at::cuda::getCurrentCUDAStream().stream();
	NMSKernel<<<
		CAFFE_GET_BLOCKS_2D(N,mask_ld),
		CAFFE_CUDA_NUM_THREADS_2D,
		0,
		stream
			>>>(
					d_boxes,
					N,
					thresh,
					mask_ld,
					dev_delete_mask.data<int>()
			   );

	// Overlapping CPU computes and D2H memcpy
	// both take about the same time
	cudaEvent_t copy_done;
	cudaEventCreate(&copy_done);
	int nto_copy = std::min(CHUNK_SIZE, N);
	cudaMemcpyAsync(&h_delete_mask[0],
			&d_delete_mask[0],
			nto_copy*mask_ld*sizeof(int),
			cudaMemcpyDeviceToHost,
			stream);
  THCudaCheck(cudaGetLastError());
	cudaEventRecord(copy_done, stream);
  THCudaCheck(cudaGetLastError());
	int offset = 0;
	std::vector<int> h_keep_sorted_list;
	std::vector<int> rmv(mask_ld, 0);
	while(offset < N) {
		const int ncopied = nto_copy;
		int next_offset = offset + ncopied;
		nto_copy = std::min(CHUNK_SIZE, N-next_offset);
		if(nto_copy > 0) {
			cudaMemcpyAsync(&h_delete_mask[next_offset*mask_ld],
					&d_delete_mask[next_offset*mask_ld],
					nto_copy*mask_ld*sizeof(int),
					cudaMemcpyDeviceToHost,
					stream);
      THCudaCheck(cudaGetLastError());
		}
		// Waiting for previous copy
		cudaEventSynchronize(copy_done);
    THCudaCheck(cudaGetLastError());

		if(nto_copy > 0) {
			cudaEventRecord(copy_done, stream);
    }

		for(int i=offset; i<next_offset; ++i) {
			int iblock = i/BOXES_PER_THREAD;
			int inblock = i%BOXES_PER_THREAD;
			if(!(rmv[iblock] & (1 << inblock))) {
				h_keep_sorted_list.push_back(i);
				int *p = &h_delete_mask[i*mask_ld];
				for(int ib=0; ib<mask_ld; ++ib) {
					rmv[ib] |= p[ib];
				}
			}
		}
		offset = next_offset;
	}
	cudaEventDestroy(copy_done);

	const int nkeep = h_keep_sorted_list.size();
  at::Tensor keep_sorted_list = at::zeros({nkeep}, torch::CUDA(at::kInt));
	cudaMemcpyAsync(keep_sorted_list.data<int>(),
			&h_keep_sorted_list[0],
			nkeep*sizeof(int),
			cudaMemcpyHostToDevice,
			stream);
  THCudaCheck(cudaGetLastError());

	// *h_nkeep = nkeep;

  // nkeep is given by the size of keep_sorted_list
  return keep_sorted_list;
}

}  // namespace utils

/**
 * This will handle the parts of Hugo's GenerateProposalsOp that
 * can't be efficiently described in pure pytorch
 */

/**
 * Generate boxes associated to topN pre-NMS scores
 */
std::vector<at::Tensor> GeneratePreNMSUprightBoxes(
        const int num_images,
        const int A,
        const int H,
        const int W,
        at::Tensor& sorted_indices, // topK sorted pre_nms_topn indices
        at::Tensor& sorted_scores,  // topK sorted pre_nms_topn scores [N, A, H, W]
        at::Tensor& bbox_deltas,    // [N, A*4, H, W] (full, unsorted / sliced)
        at::Tensor& anchors,        // input (full, unsorted, unsliced)
        at::Tensor& image_shapes,   // (h, w) of images
        const int pre_nms_nboxes,
        const int feature_stride,
        const int rpn_min_size,
        const float bbox_xform_clip_default,
        const bool correct_transform_coords) {
  // constants
  constexpr int box_dim = 4;
  const int K = H * W;
  const int conv_layer_nboxes = K * A;

  // temp Tensors
  // at::Tensor nboxes = at::zeros({num_images}, at::CUDA(at::kFloat));
  // at::Tensor boxes = at::zeros({num_images, box_dim * pre_nms_nboxes}, at::CUDA(at::kFloat));
  // at::Tensor boxes = sorted_scores.type().tensor({num_images, box_dim * pre_nms_nboxes}).to(at::kFloat);
  at::Tensor boxes = at::zeros({num_images, box_dim * pre_nms_nboxes}, sorted_scores.options()).to(at::kFloat);
  // at::Tensor boxes_keep_flags = at::zeros({num_images, box_dim * pre_nms_nboxes}, at::CUDA(at::kInt));
  // at::Tensor boxes_keep_flags = sorted_scores.type().tensor({num_images, pre_nms_nboxes}).to(at::kInt);
  at::Tensor boxes_keep_flags = at::empty({num_images, pre_nms_nboxes}, sorted_scores.options()).to(at::kInt);
  boxes_keep_flags.zero_();

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  // Call kernel
  GeneratePreNMSUprightBoxesKernel<<<
      (CAFFE_GET_BLOCKS(pre_nms_nboxes), num_images),
      CAFFE_CUDA_NUM_THREADS, // blockDim.y == 1
      0, stream>>>(
          sorted_indices.data<long>(),
          pre_nms_nboxes,
          bbox_deltas.data<float>(),
          reinterpret_cast<float4*>(anchors.data<float>()),
          H,
          W,
          K,
          A,
          K * A,
          feature_stride,
          rpn_min_size,
          image_shapes.data<float>(), // image size vec
          num_images,
          bbox_xform_clip_default, // utils::BBOX_XFORM_CLIP_DEFAULT
          correct_transform_coords,
          reinterpret_cast<float4*>(boxes.data<float>()),
          pre_nms_nboxes,
          sorted_scores.data<float>(),
          boxes_keep_flags.data<int>());
  THCudaCheck(cudaGetLastError());

  return std::vector<at::Tensor>{boxes, sorted_scores, boxes_keep_flags};
}

