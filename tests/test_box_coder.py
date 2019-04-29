# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import unittest

import numpy as np
import torch
from maskrcnn_benchmark.modeling.box_coder import BoxCoder


class TestBoxCoder(unittest.TestCase):
    def test_box_decoder(self):
        """ Match unit test UtilsBoxesTest.TestBboxTransformRandom in
            caffe2/operators/generate_proposals_op_util_boxes_test.cc
        """
        box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        bbox = torch.from_numpy(
            np.array(
                [
                    175.62031555,
                    20.91103172,
                    253.352005,
                    155.0145874,
                    169.24636841,
                    4.85241556,
                    228.8605957,
                    105.02092743,
                    181.77426147,
                    199.82876587,
                    192.88427734,
                    214.0255127,
                    174.36262512,
                    186.75761414,
                    296.19091797,
                    231.27906799,
                    22.73153877,
                    92.02596283,
                    135.5695343,
                    208.80291748,
                ]
            )
            .astype(np.float32)
            .reshape(-1, 4)
        )

        deltas = torch.from_numpy(
            np.array(
                [
                    0.47861834,
                    0.13992102,
                    0.14961673,
                    0.71495209,
                    0.29915856,
                    -0.35664671,
                    0.89018666,
                    0.70815367,
                    -0.03852064,
                    0.44466892,
                    0.49492538,
                    0.71409376,
                    0.28052918,
                    0.02184832,
                    0.65289006,
                    1.05060139,
                    -0.38172557,
                    -0.08533806,
                    -0.60335309,
                    0.79052375,
                ]
            )
            .astype(np.float32)
            .reshape(-1, 4)
        )

        gt_bbox = (
            np.array(
                [
                    206.949539,
                    -30.715202,
                    297.387665,
                    244.448486,
                    143.871216,
                    -83.342888,
                    290.502289,
                    121.053398,
                    177.430283,
                    198.666245,
                    196.295273,
                    228.703079,
                    152.251892,
                    145.431564,
                    387.215454,
                    274.594238,
                    5.062420,
                    11.040955,
                    66.328903,
                    269.686218,
                ]
            )
            .astype(np.float32)
            .reshape(-1, 4)
        )

        results = box_coder.decode(deltas, bbox)

        np.testing.assert_allclose(results.detach().numpy(), gt_bbox, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
