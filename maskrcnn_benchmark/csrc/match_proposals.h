#pragma once
#include "cuda/vision.h"
#ifndef _match_proposals_h_
#define _match_proposals_h_ 


at::Tensor match_proposals( at::Tensor match_quality_matrix, bool allow_low_quality_matches, float low_th, float high_th){
  at::Tensor result = match_proposals_cuda( match_quality_matrix, allow_low_quality_matches, low_th, high_th);
  return result;
}

#endif
