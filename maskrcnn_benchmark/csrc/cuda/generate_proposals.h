/******************************************************************************
*
* Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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
*

 ******************************************************************************/

#pragma once

#include <ATen/ATen.h>
#include <THC/THC.h>

#include <vector>

namespace utils {

// Computes Non-Maximum Suppression on the GPU
// Reject a bounding box if its region has an intersection-overunion (IoU)
//    overlap with a higher scoring selected bounding box larger than a
//    threshold.
// Out: Indices to be removed
// In:
//  * Sorted boxes
//  * Number of boxes
//  * IoU threshold
at::Tensor nms_gpu_upright(
		at::Tensor& boxes,
		const int N,
		const float thresh);
}

/**
 * Generate boxes associated to topN pre-NMS scores
 */
std::vector<at::Tensor> GeneratePreNMSUprightBoxes(
        const int num_images,
        const int A,
        const int H,
        const int W,
        at::Tensor& sorted_indices, // sorted pre_nms_topn indices
        at::Tensor& sorted_scores,  // sorted pre_nms_topn scores
        at::Tensor& bbox_deltas,    // input
        at::Tensor& anchors,        // input
        at::Tensor& image_shapes,
        const int pre_nms_nboxes,
        const int feature_stride,
        const int rpn_min_size,
        const float bbox_xform_clip_default,
        const bool correct_transform_coords);
