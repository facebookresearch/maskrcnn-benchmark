#pragma once
#include "cpu/vision.h"
#ifndef _generate_mask_targets_h_
#define _generate_mask_targets_h_ 

at::Tensor generate_mask_targets( at::Tensor dense_vector, const std::vector<std::vector<at::Tensor>> polygons, const at::Tensor anchors, const int mask_size){
  at::Tensor result = generate_mask_targets_cuda(dense_vector, polygons,anchors, mask_size);
  return result;
}

#endif

