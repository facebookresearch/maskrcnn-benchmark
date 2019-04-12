# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import unittest
import torch
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask


class TestSegmentationMask(unittest.TestCase):
    def __init__(self, method_name='runTest'):
        super(TestSegmentationMask, self).__init__(method_name)
        poly = [[[423.0, 306.5, 406.5, 277.0, 400.0, 271.5, 389.5, 277.0,
                  387.5, 292.0, 384.5, 295.0, 374.5, 220.0, 378.5, 210.0,
                  391.0, 200.5, 404.0, 199.5, 414.0, 203.5, 425.5, 221.0,
                  438.5, 297.0, 423.0, 306.5],
                 [100, 100,     200, 100,     200, 200,     100, 200],
                ]]
        width = 640
        height = 480
        size = width, height

        self.P = SegmentationMask(poly, size, 'poly')
        self.M = SegmentationMask(poly, size, 'poly').convert('mask')


    def L1(self, A, B):
        diff = A.get_mask_tensor() - B.get_mask_tensor()
        diff = torch.sum(torch.abs(diff.float())).item()
        return diff


    def test_convert(self):
        M_hat = self.M.convert('poly').convert('mask')
        P_hat = self.P.convert('mask').convert('poly')

        diff_mask = self.L1(self.M, M_hat)
        diff_poly = self.L1(self.P, P_hat)
        self.assertTrue(diff_mask == diff_poly)
        self.assertTrue(diff_mask <= 8169.)
        self.assertTrue(diff_poly <= 8169.)


    def test_crop(self):
        box = [400, 250, 500, 300] # xyxy
        diff = self.L1(self.M.crop(box), self.P.crop(box))
        self.assertTrue(diff <= 1.)


    def test_resize(self):
        new_size = 50, 25
        M_hat = self.M.resize(new_size)
        P_hat = self.P.resize(new_size)
        diff = self.L1(M_hat, P_hat)

        self.assertTrue(self.M.size == self.P.size)
        self.assertTrue(M_hat.size == P_hat.size)
        self.assertTrue(self.M.size != M_hat.size)
        self.assertTrue(diff <= 255.)


    def test_transpose(self):
        FLIP_LEFT_RIGHT = 0
        FLIP_TOP_BOTTOM = 1
        diff_hor = self.L1(self.M.transpose(FLIP_LEFT_RIGHT),
                           self.P.transpose(FLIP_LEFT_RIGHT))

        diff_ver = self.L1(self.M.transpose(FLIP_TOP_BOTTOM),
                           self.P.transpose(FLIP_TOP_BOTTOM))

        self.assertTrue(diff_hor <= 53250.)
        self.assertTrue(diff_ver <= 42494.)


if __name__ == "__main__":

    unittest.main()
