import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from maskrcnn_benchmark import _Custom as _C

class _HoughVotingFunction(Function):
    @staticmethod
    def forward(ctx, labels, masks, vertex_pred, extents, poses, meta_data, 
            inlier_threshold, skip_pixels):
        # ctx.save_for_backward(masks, vertex_pred)

        if masks.is_cuda:
            masks_int = masks.type(torch.cuda.IntTensor) if masks.type() != 'torch.cuda.IntTensor' else masks
            labels_int = labels.type(torch.cuda.IntTensor) if labels.type() != 'torch.cuda.IntTensor' else labels
            output = _C.hough_voting_forward(labels_int, masks_int, vertex_pred, extents, meta_data, poses, 
                        inlier_threshold, skip_pixels)
        else:
            raise NotImplementedError("Hough Voting Forward CPU layer not implemented!")

        top_box, top_pose = output  # MUST UNROLL THIS AND RETURN
        return top_box, top_pose

class HoughVoting(nn.Module):
    def __init__(self, inlier_threshold=0.9, skip_pixels=1):
        super(HoughVoting, self).__init__()

        self.inlier_threshold = float(inlier_threshold)
        self.skip_pixels = int(skip_pixels)

    def forward(self, labels, masks, vertex_pred, extents, poses, meta_data):
        return _HoughVotingFunction.apply(labels, masks, vertex_pred, extents, poses, meta_data, 
                    self.inlier_threshold, self.skip_pixels)

    def __repr__(self):
        tmpstr = "%s (inlier_threshold=%.2f, skip_pixels=%d)"%(self.__class__.__name__, 
                    self.inlier_threshold, self.skip_pixels)
        return tmpstr
