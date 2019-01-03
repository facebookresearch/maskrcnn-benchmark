#pragma once
#include "cuda/vision.h"
#ifndef _box_encode_h_
#define _box_encode_h_ 

std::vector<at::Tensor> box_encode(at::Tensor boxes, at::Tensor anchors, float wx, float wy, float ww, float wh){
  std::vector<at::Tensor> result = box_encode_cuda(boxes, anchors, wx, wy, ww, wh);
  return result;
}

#endif
