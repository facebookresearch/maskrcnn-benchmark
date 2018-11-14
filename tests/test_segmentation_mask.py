# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import numpy as np
from common_utils import TestCase, run_tests
from maskrcnn_benchmark.structures.segmentation_mask import Mask, Polygons, SegmentationMask


class TestSegmentationMask(TestCase):
    def __init__(self, method_name='runTest'):
        super(TestSegmentationMask, self).__init__(method_name)
        self.poly = [[423.0, 306.5, 406.5, 277.0, 400.0, 271.5, 389.5, 277.0, 387.5, 292.0,
                    384.5, 295.0, 374.5, 220.0, 378.5, 210.0, 391.0, 200.5, 404.0, 199.5,
                    414.0, 203.5, 425.5, 221.0, 438.5, 297.0, 423.0, 306.5],
                   [385.5, 240.0, 404.0, 234.5, 419.5, 234.0, 416.5, 219.0, 409.0, 209.5,
                    394.0, 207.5, 385.5, 213.0, 382.5, 221.0, 385.5, 240.0]]
        self.width = 640
        self.height = 480
        self.size = (self.width, self.height)
        self.box = [35, 55, 540, 400] # xyxy

        self.polygon = Polygons(self.poly, self.size, 'polygon')
        self.mask = Mask(self.poly, self.size, 'mask')

    def test_crop(self):
        poly_crop = self.polygon.crop(self.box)
        mask_from_poly_crop = poly_crop.convert('mask')
        mask_crop = self.mask.crop(self.box).mask

        self.assertEqual(mask_from_poly_crop, mask_crop)
    
    def test_convert(self):
        mask_from_poly_convert = self.polygon.convert('mask')
        mask = self.mask.convert('mask')
        self.assertEqual(mask_from_poly_convert, mask)

    def test_transpose(self):
        FLIP_LEFT_RIGHT = 0
        FLIP_TOP_BOTTOM = 1
        methods = (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM)
        for method in methods:
            mask_from_poly_flip = self.polygon.transpose(method).convert('mask')
            mask_flip = self.mask.transpose(method).convert('mask')
            print(method, torch.abs(mask_flip.float() - mask_from_poly_flip.float()).sum())
            self.assertEqual(mask_flip, mask_from_poly_flip)
    
    def test_resize(self):
        new_size = (600, 500)
        mask_from_poly_resize = self.polygon.resize(new_size).convert('mask')
        mask_resize = self.mask.resize(new_size).convert('mask')
        print('diff resize: ', torch.abs(mask_from_poly_resize.float() - mask_resize.float()).sum())
        self.assertEqual(mask_from_poly_resize, mask_resize)

if __name__ == "__main__":
    run_tests()
