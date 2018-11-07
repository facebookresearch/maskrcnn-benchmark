# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

import numpy as np
import torch
from torch import nn

from ..structures.bounding_box import BoxList
from .utils import meshgrid

import maskrcnn_benchmark._C as C

# meshgrid = torch.meshgrid


class AnchorGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set
    of anchors
    """

    def __init__(
        self,
        scales=(0.5, 1.0, 2.0),
        aspect_ratios=(0.5, 1.0, 2.0),
        base_anchor_size=256,
        anchor_stride=16,
        straddle_thresh=0,
        *args,
        **kwargs
    ):
        super(AnchorGenerator, self).__init__()
        # TODO complete and fix
        sizes = tuple(i * base_anchor_size for i in scales)

        cell_anchors = generate_anchors(anchor_stride, sizes, aspect_ratios).float()
        self.stride = anchor_stride
        self.cell_anchors = cell_anchors
        self.straddle_thresh = straddle_thresh

    def num_anchors_per_location(self):
        return [len(self.cell_anchors)]

    def forward_single_image(self, image_sizes, feature_map):
        device = feature_map.device
        grid_height, grid_width = feature_map.shape[-2:]
        stride = self.stride
        shifts_x = torch.arange(
            0, grid_width * stride, step=stride, dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, grid_height * stride, step=stride, dtype=torch.float32, device=device
        )
        shift_x, shift_y = meshgrid(shifts_x, shifts_y)
        shift_x = shift_x.view(-1)
        shift_y = shift_y.view(-1)
        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

        self.cell_anchors = self.cell_anchors.to(device)

        A = self.cell_anchors.size(0)
        K = shifts.size(0)
        field_of_anchors = self.cell_anchors.view(1, A, 4) + shifts.view(
            1, K, 4
        ).permute(1, 0, 2)
        anchors = field_of_anchors.view(K * A, 4)

        # add visibility information to anchors
        image_height, image_width = image_sizes

        if self.straddle_thresh >= 0:
            inds_inside = (
                (anchors[..., 0] >= -self.straddle_thresh)
                & (anchors[..., 1] >= -self.straddle_thresh)
                & (anchors[..., 2] < image_width + self.straddle_thresh)
                & (anchors[..., 3] < image_height + self.straddle_thresh)
            )
        else:
            inds_inside = torch.ones(anchors.shape[0], dtype=torch.uint8, device=device)

        anchors = BoxList(anchors, (image_width, image_height), mode="xyxy")
        anchors.add_field("visibility", inds_inside)

        # TODO check if want to return list of not
        # return [anchors]
        return anchors

    def forward(self, images_sizes, feature_maps):
        """
        Arguments:
            image_sizes (list(tuple(int, int)))
            feature_maps (list(list(tensor)))
        """
        assert len(feature_maps) == 1, "only single feature maps allowed"
        anchors = []
        for image_sizes, feature_map in zip(images_sizes, feature_maps[0]):
            anchors.append(self.forward_single_image(image_sizes, feature_map))
        return [anchors]


# TODO deduplicate with AnchorGenerator
class FPNAnchorGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set
    of anchors
    """

    def __init__(
        self,
        scales=(0.5, 1.0, 2.0),
        aspect_ratios=(0.5, 1.0, 2.0),
        base_anchor_size=256,
        anchor_strides=(8, 16, 32),
        straddle_thresh=0,
        *args,
        **kwargs
    ):
        super(FPNAnchorGenerator, self).__init__()
        # TODO complete and fix
        sizes = tuple(i * base_anchor_size for i in scales)

        cell_anchors = [
            generate_anchors(anchor_stride, (size,), aspect_ratios).float()
            for anchor_stride, size in zip(anchor_strides, sizes)
        ]
        self.strides = anchor_strides
        self.cell_anchors = cell_anchors
        self.straddle_thresh = straddle_thresh

    def num_anchors_per_location(self):
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def forward_single_image(self, image_sizes, feature_map, cell_anchors, stride):
        device = feature_map.device
        grid_height, grid_width = feature_map.shape[-2:]
        # add visibility information to anchors
        image_height, image_width = image_sizes
        # Replace chunk with a single kernel call
        if False:
            shifts_x = torch.arange(
                0, grid_width * stride, step=stride, dtype=torch.float32, device=device
            )
            shifts_y = torch.arange(
                0, grid_height * stride, step=stride, dtype=torch.float32, device=device
            )
            shift_x, shift_y = meshgrid(shifts_x, shifts_y)
            shift_x = shift_x.view(-1)
            shift_y = shift_y.view(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            A = cell_anchors.size(0)
            K = shifts.size(0)
            field_of_anchors = cell_anchors.view(1, A, 4) + shifts.view(1, K, 4).permute(
                1, 0, 2
            )
            anchors = field_of_anchors.view(K * A, 4)


            if self.straddle_thresh >= 0:
                inds_inside = (
                    (anchors[..., 0] >= -self.straddle_thresh)
                    & (anchors[..., 1] >= -self.straddle_thresh)
                    & (anchors[..., 2] < image_width + self.straddle_thresh)
                    & (anchors[..., 3] < image_height + self.straddle_thresh)
                )
            else:
                inds_inside = torch.ones(anchors.shape[0], dtype=torch.uint8, device=device)
        else:
            anchors, inds_inside = C.anchor_generator(image_sizes, (grid_height, grid_width), cell_anchors, stride, self.straddle_thresh)

        anchors = BoxList(anchors, (image_width, image_height), mode="xyxy")
        anchors.add_field("visibility", inds_inside)

        # TODO check if want to return list of not
        # return [anchors]
        return anchors

    def forward(self, images_sizes, feature_maps):
        """
        Arguments:
            image_sizes (list(tuple(int, int)))
            feature_maps (list(list(tensor)))
        """
        device = feature_maps[0][0].device
        self.cell_anchors = [anchor.to(device) for anchor in self.cell_anchors]
        anchors = []
        for feature_map_level, stride, cell_anchor in zip(
            feature_maps, self.strides, self.cell_anchors
        ):
            per_level_anchors = []
            for image_sizes, feature_map in zip(images_sizes, feature_map_level):
                per_level_anchors.append(
                    self.forward_single_image(
                        image_sizes, feature_map, cell_anchor, stride
                    )
                )
            anchors.append(per_level_anchors)
        return anchors


# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------


# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

# array([[ -83.,  -39.,  100.,   56.],
#        [-175.,  -87.,  192.,  104.],
#        [-359., -183.,  376.,  200.],
#        [ -55.,  -55.,   72.,   72.],
#        [-119., -119.,  136.,  136.],
#        [-247., -247.,  264.,  264.],
#        [ -35.,  -79.,   52.,   96.],
#        [ -79., -167.,   96.,  184.],
#        [-167., -343.,  184.,  360.]])


def generate_anchors(
    stride=16, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)
):
    """Generates a matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    are centered on stride / 2, have (approximate) sqrt areas of the specified
    sizes, and aspect ratios as given.
    """
    return _generate_anchors(
        stride,
        np.array(sizes, dtype=np.float) / stride,
        np.array(aspect_ratios, dtype=np.float),
    )


def _generate_anchors(base_size, scales, aspect_ratios):
    """Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, base_size - 1, base_size - 1) window.
    """
    anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 1
    anchors = _ratio_enum(anchor, aspect_ratios)
    anchors = np.vstack(
        [_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
    )
    return torch.from_numpy(anchors)


def _whctrs(anchor):
    """Return width, height, x center, and y center for an anchor (window)."""
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack(
        (
            x_ctr - 0.5 * (ws - 1),
            y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1),
            y_ctr + 0.5 * (hs - 1),
        )
    )
    return anchors


def _ratio_enum(anchor, ratios):
    """Enumerate a set of anchors for each aspect ratio wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """Enumerate a set of anchors for each scale wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


# TODO doesn't match exactly Detectron
# heavily inspired from tensorflow
class AnchorGenerator_v0(nn.Module):
    def __init__(
        self,
        scales=(0.5, 1.0, 2.0),
        aspect_ratios=(0.5, 1.0, 2.0),
        base_anchor_size=None,
        anchor_stride=None,
        anchor_offset=None,
        straddle_thresh=0,
    ):
        super(AnchorGenerator, self).__init__()
        # Handle argument defaults
        if base_anchor_size is None:
            base_anchor_size = [256, 256]
        base_anchor_size = torch.tensor(base_anchor_size, dtype=torch.float32)
        if anchor_stride is None:
            anchor_stride = [16, 16]
        anchor_stride = torch.tensor(anchor_stride, dtype=torch.float32)
        if anchor_offset is None:
            anchor_offset = [0, 0]
        anchor_offset = torch.tensor(anchor_offset, dtype=torch.float32)

        scales = torch.tensor(scales, dtype=torch.float32)
        aspect_ratios = torch.tensor(aspect_ratios, dtype=torch.float32)

        self.register_buffer("_scales", scales)
        self.register_buffer("_aspect_ratios", aspect_ratios)
        self.register_buffer("_base_anchor_size", base_anchor_size)
        self.register_buffer("_anchor_stride", anchor_stride)
        self.register_buffer("_anchor_offset", anchor_offset)

        """
        self._scales = scales
        self._aspect_ratios = aspect_ratios
        self._base_anchor_size = base_anchor_size
        self._anchor_stride = anchor_stride
        self._anchor_offset = anchor_offset
        """
        self.straddle_thresh = straddle_thresh

    def num_anchors_per_location(self):
        return [len(self._scales) * len(self._aspect_ratios)]

    # TODO we don't want image, but image shapes. We can't get individual
    # images like this -- only relevant for the visibility
    def forward_single_image(self, image_sizes, feature_map):
        """
        only the first element of the list is taken into account
        image is a list of tensors
        feature_map is a list of tensors
        """
        # TODO attention if we want to return a list or not
        # grid_height, grid_width = feature_map[0].shape[-2:]
        grid_height, grid_width = feature_map.shape[-2:]
        scales_grid, aspect_ratios_grid = meshgrid(self._scales, self._aspect_ratios)
        scales_grid = torch.reshape(scales_grid, [-1])
        aspect_ratios_grid = torch.reshape(aspect_ratios_grid, [-1])

        # for the JIT  -- JIT doesn't work here
        # grid_height = torch.tensor(grid_height)
        # grid_width = torch.tensor(grid_width)

        anchors = tile_anchors(
            grid_height,
            grid_width,
            scales_grid,
            aspect_ratios_grid,
            self._base_anchor_size,
            self._anchor_stride,
            self._anchor_offset,
        )

        # add visibility information to anchors
        image_height, image_width = image_sizes
        inds_inside = (
            (anchors[..., 0] >= -self.straddle_thresh)
            & (anchors[..., 1] >= -self.straddle_thresh)
            & (anchors[..., 2] < image_width + self.straddle_thresh)
            & (anchors[..., 3] < image_height + self.straddle_thresh)
        )

        anchors = BoxList(anchors, (image_width, image_height), mode="xyxy")
        anchors.add_field("visibility", inds_inside)

        # TODO check if want to return list of not
        # return [anchors]
        return anchors

    def forward(self, images_sizes, feature_maps):
        anchors = []
        for image_sizes, feature_map in zip(images_sizes, feature_maps):
            anchors.append(self.forward_single_image(image_sizes, feature_map))
        return anchors


# copyied from tensorflow
# @torch.jit.compile(nderivs=0)  # TODO JIT doesn't work
def tile_anchors(
    grid_height,
    grid_width,
    scales,
    aspect_ratios,
    base_anchor_size,
    anchor_stride,
    anchor_offset,
):

    device = scales.device

    ratio_sqrts = torch.sqrt(aspect_ratios)
    heights = scales / ratio_sqrts * base_anchor_size[0]
    widths = scales * ratio_sqrts * base_anchor_size[1]

    # heights = torch.round((widths - 1) / aspect_ratios + 1)
    # widths = torch.round((heights - 1) * aspect_ratios + 1)
    print(heights, widths)

    # TODO extra here
    # heights = heights.round()
    # widths = widths.round()

    # TODO it seems that cuda arange is much slower?
    # TODO replace scale + shift with a single call to arange

    # Get a grid of box centers
    y_centers = torch.arange(grid_height, dtype=torch.float32, device=device)
    y_centers = y_centers * anchor_stride[0] + anchor_offset[0]
    x_centers = torch.arange(grid_width, dtype=torch.float32, device=device)
    x_centers = x_centers * anchor_stride[1] + anchor_offset[1]
    x_centers, y_centers = meshgrid(x_centers, y_centers)

    widths_grid, x_centers_grid = meshgrid(widths, x_centers)
    heights_grid, y_centers_grid = meshgrid(heights, y_centers)
    bbox_centers = torch.stack([y_centers_grid, x_centers_grid], dim=3)
    bbox_sizes = torch.stack([heights_grid, widths_grid], dim=3)
    bbox_centers = torch.reshape(bbox_centers, [-1, 2])
    bbox_sizes = torch.reshape(bbox_sizes, [-1, 2])
    # bbox_corners = torch.cat([bbox_centers - .5 * bbox_sizes, bbox_centers + .5 * bbox_sizes], 1)
    bbox_corners = torch.cat(
        [bbox_centers - .5 * (bbox_sizes - 1), bbox_centers + .5 * (bbox_sizes - 1)], 1
    )
    return bbox_corners


if __name__ == "__main__":
    g = AnchorGenerator(anchor_offset=(8.5, 8.5))
    # g = AnchorGenerator(anchor_offset=(9., 9.))
    tt = g([[10, 10]], [torch.rand(1, 3, 1, 1)])

    t = torch.tensor(
        [
            [-83., -39., 100., 56.],
            [-175., -87., 192., 104.],
            [-359., -183., 376., 200.],
            [-55., -55., 72., 72.],
            [-119., -119., 136., 136.],
            [-247., -247., 264., 264.],
            [-35., -79., 52., 96.],
            [-79., -167., 96., 184.],
            [-167., -343., 184., 360.],
        ]
    )

    print(t - tt[0].bbox)
    # from IPython import embed; embed()
