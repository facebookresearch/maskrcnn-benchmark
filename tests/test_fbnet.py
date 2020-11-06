# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import unittest

import numpy as np
import torch
import maskrcnn_benchmark.modeling.backbone.fbnet_builder as fbnet_builder


TEST_CUDA = torch.cuda.is_available()


def _test_primitive(self, device, op_name, op_func, N, C_in, C_out, expand, stride):
    op = op_func(C_in, C_out, expand, stride).to(device)
    input = torch.rand([N, C_in, 7, 7], dtype=torch.float32).to(device)
    output = op(input)
    self.assertEqual(
        output.shape[:2], torch.Size([N, C_out]),
        'Primitive {} failed for shape {}.'.format(op_name, input.shape)
    )


class TestFBNetBuilder(unittest.TestCase):
    def test_identity(self):
        id_op = fbnet_builder.Identity(20, 20, 1)
        input = torch.rand([10, 20, 7, 7], dtype=torch.float32)
        output = id_op(input)
        np.testing.assert_array_equal(np.array(input), np.array(output))

        id_op = fbnet_builder.Identity(20, 40, 2)
        input = torch.rand([10, 20, 7, 7], dtype=torch.float32)
        output = id_op(input)
        np.testing.assert_array_equal(output.shape, [10, 40, 4, 4])

    def test_primitives(self):
        ''' Make sures the primitives runs '''
        for op_name, op_func in fbnet_builder.PRIMITIVES.items():
            print('Testing {}'.format(op_name))

            _test_primitive(
                self, "cpu",
                op_name, op_func,
                N=20, C_in=16, C_out=32, expand=4, stride=1
            )

    @unittest.skipIf(not TEST_CUDA, "no CUDA detected")
    def test_primitives_cuda(self):
        ''' Make sures the primitives runs on cuda '''
        for op_name, op_func in fbnet_builder.PRIMITIVES.items():
            print('Testing {}'.format(op_name))

            _test_primitive(
                self, "cuda",
                op_name, op_func,
                N=20, C_in=16, C_out=32, expand=4, stride=1
            )

    def test_primitives_empty_batch(self):
        ''' Make sures the primitives runs '''
        for op_name, op_func in fbnet_builder.PRIMITIVES.items():
            print('Testing {}'.format(op_name))

            # test empty batch size
            _test_primitive(
                self, "cpu",
                op_name, op_func,
                N=0, C_in=16, C_out=32, expand=4, stride=1
            )

    @unittest.skipIf(not TEST_CUDA, "no CUDA detected")
    def test_primitives_cuda_empty_batch(self):
        ''' Make sures the primitives runs '''
        for op_name, op_func in fbnet_builder.PRIMITIVES.items():
            print('Testing {}'.format(op_name))

            # test empty batch size
            _test_primitive(
                self, "cuda",
                op_name, op_func,
                N=0, C_in=16, C_out=32, expand=4, stride=1
            )

if __name__ == "__main__":
    unittest.main()
