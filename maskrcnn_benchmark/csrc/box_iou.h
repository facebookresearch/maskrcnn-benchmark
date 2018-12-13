#pragma once
#include "cuda/vision.h"
#ifndef _box_iou_h_
#define _box_iou_h_ 

at::Tensor box_iou(at::Tensor box1, at::Tensor box2){
  at::Tensor result = box_iou_cuda(box1, box2);
  return result;
}

#endif

