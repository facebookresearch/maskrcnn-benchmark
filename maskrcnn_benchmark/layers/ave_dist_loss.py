import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from maskrcnn_benchmark import _Custom as _C


class _AverageDistanceLossFunction(Function):
    @staticmethod
    def forward(ctx, poses_pred, poses_target, poses_labels, points, symmetry, margin):
        assert points.size(0) == symmetry.size(0)
        assert poses_pred.size(-1) == 4
        assert poses_pred.size() == poses_target.size()
        assert poses_pred.size(0) == poses_labels.size(0)

        if poses_labels.is_cuda:
            poses_labels_int = poses_labels.type(torch.cuda.IntTensor) if poses_labels.type() != 'torch.cuda.IntTensor' else poses_labels
            outputs = _C.average_distance_loss_forward(poses_pred, poses_target, poses_labels_int, points, symmetry, margin)
        else:
            raise NotImplementedError("Average Distance Loss Forward CPU layer not implemented!")
        
        loss, bottom_diff = outputs
        ctx.save_for_backward(bottom_diff)

        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        bottom_diff, = ctx.saved_tensors
        # grad_input = bottom_diff.new(*bottom_diff.size()).zero_()
        grad_input = _C.average_distance_loss_backward(grad, bottom_diff);

        """
        gradients for: poses_pred, poses_target, poses_labels, points, symmetry, margin
        """
        return grad_input, None, None, None, None, None


class AverageDistanceLoss(nn.Module):
    def __init__(self, margin):
        super(AverageDistanceLoss, self).__init__()

        self.margin = float(margin)

    def forward(self, poses_pred, poses_target, poses_labels, points, symmetry):
        return _AverageDistanceLossFunction.apply(poses_pred, poses_target, poses_labels, points, symmetry, self.margin)

    def __repr__(self):
        tmpstr = "%s (margin=%.3f)"%(self.__class__.__name__, self.margin)
        return tmpstr
