# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
Miscellaneous utility functions
"""

import logging

import torch

from ..structures.bounding_box import BoxList


def nonzero(tensor):
    """
    Equivalent to numpy.nonzero(). The difference torch.nonzero()
    is that it returns a tuple of 1d tensors, and not a 2d tensor.
    This is more convenient in a few cases.
    This should maybe be sent to core pytorch
    """
    result = tensor.nonzero()
    return torch.unbind(result, 1)


# TODO maybe push this to nn?
def smooth_l1_loss(input, target, beta=1. / 9, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()


def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def cat_bbox(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes (list of BoxList)
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in bboxes)

    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)

    cat_boxes = BoxList(cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)

    for field in fields:
        data = cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes


def split_bbox(bbox, split_size_or_sections):
    assert isinstance(bbox, BoxList)

    boxes = bbox.bbox.split(split_size_or_sections, dim=0)
    size = bbox.size
    mode = bbox.mode
    fields = bbox.fields()
    fields_data = {
        field: bbox.get_field(field).split(split_size_or_sections) for field in fields
    }
    bboxes = [BoxList(box, size, mode) for box in boxes]
    for i, box in enumerate(bboxes):
        [
            box.add_field(field, field_data[i])
            for field, field_data in fields_data.items()
        ]

    return bboxes


def split_with_sizes(tensor, split_sizes, dim=0):
    """
    Similar to split, but with a fix for
    https://github.com/pytorch/pytorch/issues/8686
    """
    assert sum(split_sizes) == tensor.shape[dim]
    result = []
    start_idx = 0
    for length in split_sizes:
        if length == 0:
            result.append(tensor.new())
            continue
        result.append(tensor.narrow(dim, start_idx, length))
        start_idx += length
    assert start_idx == tensor.shape[dim]
    return result


def keep_only_positive_boxes(boxes):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], (list, tuple))
    assert isinstance(boxes[0][0], BoxList)
    assert boxes[0][0].has_field("labels")
    positive_boxes = []
    num_boxes = 0
    for boxes_per_feature_map in boxes:
        positive_boxes_per_feature_map = []
        for boxes_per_image in boxes_per_feature_map:
            labels = boxes_per_image.get_field("labels")
            inds = nonzero(labels > 0)[0]
            selected_boxes = boxes_per_image[inds]
            positive_boxes_per_feature_map.append(selected_boxes)
            num_boxes += len(inds)
        positive_boxes.append(positive_boxes_per_feature_map)
    return positive_boxes


def meshgrid(x, y=None):
    if y is None:
        y = x
    x = torch.tensor(x)
    y = torch.tensor(y)
    m, n = x.size(0), y.size(0)
    grid_x = x[None].expand(n, m).contiguous()
    grid_y = y[:, None].expand(n, m).contiguous()
    return grid_x, grid_y


def meshgrid_new(x, y):
    """
    Not used. Should eventually replace meshgrid,
    but it requires some testing
    """
    x = torch.tensor(x)
    y = torch.tensor(y)
    x_exp_shape = tuple(1 for _ in y.shape) + x.shape
    y_exp_shape = y.shape + tuple(1 for _ in x.shape)

    xgrid = x.reshape(x_exp_shape).repeat(y_exp_shape)
    ygrid = y.reshape(y_exp_shape).repeat(x_exp_shape)
    new_shape = y.shape + x.shape
    xgrid = xgrid.reshape(new_shape)
    ygrid = ygrid.reshape(new_shape)

    return xgrid, ygrid


def load_state_dict(model, state_dict, strict=True):
    """
    Very similar to model.load_state_dict, but which logs
    as well which parameters have been loaded
    """
    # perform the loading
    model.load_state_dict(state_dict, strict)
    logger = logging.getLogger("maskrcnn_benchmark.core.utils.load_state_dict")
    # get the names of the parameters that were loaded
    local_state_dict_keys = model.state_dict().keys()
    state_dict_keys = state_dict.keys()
    loaded_keys = [k for k in local_state_dict_keys if k in state_dict_keys]
    max_size = max([len(key) for key in loaded_keys]) if loaded_keys else 1
    log_str_template = "{: <{}} loaded from weights file: {}"
    for key in sorted(loaded_keys):
        logger.debug(
            log_str_template.format(key, max_size, tuple(state_dict[key].shape))
        )

class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)
    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None

# TODO move those to core pytorch
# helper class that supports empty tensors
class Conv2d(torch.nn.Conv2d):
    def forward(self, x):
        if x.numel() > 0:
            return super(Conv2d, self).forward(x)
        # get output shape

        output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // d + 1
                for i, p, di, k, d in zip(
                    x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride)]
        output_shape = [x.shape[0], self.bias.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)

class ConvTranspose2d(torch.nn.ConvTranspose2d):
    def forward(self, x):
        if x.numel() > 0:
            return super(ConvTranspose2d, self).forward(x)
        # get output shape

        output_shape = [
                (i - 1) * d - 2 * p + (di * (k - 1) + 1) + op
                for i, p, di, k, d, op in zip(
                    x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride, self.output_padding)]
        output_shape = [x.shape[0], self.bias.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)

def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    if input.numel() > 0:
        return torch.nn.functional.interpolate(input, size, scale_factor, mode, align_corners)

    from torch.nn.modules.utils import _ntuple
    import math

    def _check_size_scale_factor(dim):
        if size is None and scale_factor is None:
            raise ValueError('either size or scale_factor should be defined')
        if size is not None and scale_factor is not None:
            raise ValueError('only one of size or scale_factor should be defined')
        if scale_factor is not None and isinstance(scale_factor, tuple)\
                and len(scale_factor) != dim:
            raise ValueError('scale_factor shape must match input shape. '
                             'Input is {}D, scale_factor size is {}'.format(dim, len(scale_factor)))

    def _output_size(dim):
        _check_size_scale_factor(dim)
        if size is not None:
            return size
        scale_factors = _ntuple(dim)(scale_factor)
        # math.floor might return float in py2.7
        return [int(math.floor(input.size(i + 2) * scale_factors[i])) for i in range(dim)]

    output_shape = tuple(_output_size(2))
    output_shape = input.shape[:-2] + output_shape
    return _NewEmptyTensorOp.apply(input, output_shape)
