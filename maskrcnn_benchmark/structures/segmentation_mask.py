# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import numpy as np
from torch.nn.functional import interpolate
import pycocotools.mask as mask_utils

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1

class Mask(object):
    """
    This class is unfinished and not meant for use yet
    It is supposed to contain the mask for an object as
    a 2d tensor
    """
    def __init__(self, segm, size, mode):
        width, height = size
        if isinstance(segm, Mask):
            mask = segm.mask
        else:
            if type(segm) == list:
                # polygons
                rle = mask_utils.frPyObjects(segm, height, width)
                mask = np.array(mask_utils.decode(rle), dtype=np.float32)
                mask = np.sum(mask, axis=2)
                mask = torch.from_numpy(np.array(mask > 0, dtype=np.float32))
            elif type(segm) == dict and 'counts' in segm:
                if type(segm['counts']) == list:
                    # uncompressed RLE
                    h, w = segm['size']
                    rle = mask_utils.frPyObjects(segm, h, w)
                    mask = mask_utils.decode(rle)
                    mask = torch.from_numpy(mask).to(dtype=torch.float32)
                else:
                    # compressed RLE
                    mask = mask_utils.decode(segm)
                    mask = torch.from_numpy(mask).to(dtype=torch.float32)
            else:
                # binary mask
                if type(segm) == np.ndarray:
                    mask = torch.from_numpy(segm).to(dtype=torch.float32)
                else: # torch.Tensor
                    mask = segm
        self.mask = mask
        self.size = size
        self.mode = mode

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError("Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented")

        width, height = self.size
        if method == FLIP_LEFT_RIGHT:
            max_idx = width
            dim = 1
        elif method == FLIP_TOP_BOTTOM:
            max_idx = height
            dim = 0

        flip_idx = torch.tensor(list(range(max_idx)[::-1]))
        flipped_mask = self.mask.index_select(dim, flip_idx)
        return Mask(flipped_mask, self.size, self.mode)

    def crop(self, box):
        box = [int(b) for b in box]
        TO_REMOVE = 1
        w, h = box[2] - box[0] + TO_REMOVE, box[3] - box[1] + TO_REMOVE
        cropped_mask = self.mask[box[1]: box[3]+1, box[0]: box[2]+1]
        return Mask(cropped_mask, size=(w, h), mode=self.mode)

    # torch.nn.functional.interpolate has a arg as dim, only tensor have dim, so turn array to tensor
    def resize(self, size, *args, **kwargs):
        width, height = size
        scaled_mask = torch.squeeze(interpolate(torch.from_numpy(np.array(self.mask)[None, None, :, :]).float(),
                                                (height, width),
                                                mode='nearest'))
        return Mask(scaled_mask, size=size, mode=self.mode)

    def convert(self, mode):
        mask = self.mask
        return mask

    def __iter__(self):
        return iter(self.mask)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        # s += "num_mask={}, ".format(len(self.mask))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s



class Polygons(object):
    """
    This class holds a set of polygons that represents a single instance
    of an object mask. The object can be represented as a set of
    polygons
    """

    def __init__(self, polygons, size, mode):
        # assert isinstance(polygons, list), '{}'.format(polygons)
        if isinstance(polygons, list):
            polygons = [torch.as_tensor(p, dtype=torch.float32) for p in polygons]
        elif isinstance(polygons, Polygons):
            polygons = polygons.polygons

        self.polygons = polygons
        self.size = size
        self.mode = mode

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        flipped_polygons = []
        width, height = self.size
        if method == FLIP_LEFT_RIGHT:
            dim = width
            idx = 0
        elif method == FLIP_TOP_BOTTOM:
            dim = height
            idx = 1

        for poly in self.polygons:
            p = poly.clone()
            TO_REMOVE = 1
            p[idx::2] = dim - poly[idx::2] - TO_REMOVE
            flipped_polygons.append(p)

        return Polygons(flipped_polygons, size=self.size, mode=self.mode)

    def crop(self, box):
        TO_REMOVE = 1
        w, h = box[2] - box[0] + TO_REMOVE, box[3] - box[1] + TO_REMOVE

        # TODO chck if necessary
        w = max(w, 1)
        h = max(h, 1)

        cropped_polygons = []
        for poly in self.polygons:
            p = poly.clone()
            p[0::2] = p[0::2] - box[0]  # .clamp(min=0, max=w)
            p[1::2] = p[1::2] - box[1]  # .clamp(min=0, max=h)
            cropped_polygons.append(p)

        return Polygons(cropped_polygons, size=(w, h), mode=self.mode)

    def resize(self, size, *args, **kwargs):
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled_polys = [p * ratio for p in self.polygons]
            return Polygons(scaled_polys, size, mode=self.mode)

        ratio_w, ratio_h = ratios
        scaled_polygons = []
        for poly in self.polygons:
            p = poly.clone()
            p[0::2] *= ratio_w
            p[1::2] *= ratio_h
            scaled_polygons.append(p)

        return Polygons(scaled_polygons, size=size, mode=self.mode)

    def convert(self, mode):
        width, height = self.size
        if mode == "mask":
            rles = mask_utils.frPyObjects(
                [p.numpy() for p in self.polygons], height, width
            )
            rle = mask_utils.merge(rles)
            mask = mask_utils.decode(rle)
            mask = np.sum(mask, axis=2)
            mask = torch.from_numpy(np.array(mask > 0, dtype=np.float32))
            # TODO add squeeze?
            return mask

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_polygons={}, ".format(len(self.polygons))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s


class SegmentationMask(object):
    """
    This class stores the segmentations for all objects in the image
    """

    def __init__(self, segms, size, mode=None):
        """
        Arguments:
            segms: three types
                (1) polygons: a list of list of lists of numbers. The first
                level of the list correspond to individual instances,
                the second level to all the polygons that compose the
                object, and the third level to the polygon coordinates.
                (2) rles: COCO's run length encoding format, uncompressed or compressed
                (3) binary masks
            size: (width, height)
            mode: 'polygon', 'mask'. if mode is 'mask', convert mask of any format to binary mask
        """
        assert isinstance(segms, list)
        if type(segms[0]) != list:
            mode = 'mask'
        if mode == 'mask':
            self.masks = [Mask(m, size, mode) for m in segms]
        else: # polygons
            self.masks = [Polygons(p, size, mode) for p in segms]
        self.size = size
        self.mode = mode

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        flipped = []
        for mask in self.masks:
            flipped.append(mask.transpose(method))
        return SegmentationMask(flipped, size=self.size, mode=self.mode)

    def crop(self, box):
        w, h = box[2] - box[0] + 1, box[3] - box[1] + 1
        cropped = []
        for mask in self.masks:
            cropped.append(mask.crop(box))
        return SegmentationMask(cropped, size=(w, h), mode=self.mode)

    def resize(self, size, *args, **kwargs):
        scaled = []
        for mask in self.masks:
            scaled.append(mask.resize(size, *args, **kwargs))
        return SegmentationMask(scaled, size=size, mode=self.mode)

    def to(self, *args, **kwargs):
        return self

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            selected_masks = [self.masks[item]]
        else:
            # advanced indexing on a single dimension
            selected_masks = []
            if isinstance(item, torch.Tensor) and item.dtype == torch.uint8:
                item = item.nonzero()
                item = item.squeeze(1) if item.numel() > 0 else item
                item = item.tolist()
            for i in item:
                selected_masks.append(self.masks[i])
        return SegmentationMask(selected_masks, size=self.size, mode=self.mode)

    def __iter__(self):
        return iter(self.masks)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self.masks))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={})".format(self.size[1])
        return s
