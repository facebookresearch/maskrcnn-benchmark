# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import unittest
import copy
import torch
# import modules to to register rpn heads
from maskrcnn_benchmark.modeling.backbone import build_backbone # NoQA
from maskrcnn_benchmark.modeling.rpn.rpn import build_rpn # NoQA
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.config import cfg as g_cfg
from utils import load_config


# overwrite configs if specified, otherwise default config is used
RPN_CFGS = {
}


class TestRPNHeads(unittest.TestCase):
    def test_build_rpn_heads(self):
        ''' Make sure rpn heads run '''

        self.assertGreater(len(registry.RPN_HEADS), 0)

        in_channels = 64
        num_anchors = 10

        for name, builder in registry.RPN_HEADS.items():
            print('Testing {}...'.format(name))
            if name in RPN_CFGS:
                cfg = load_config(RPN_CFGS[name])
            else:
                # Use default config if config file is not specified
                cfg = copy.deepcopy(g_cfg)

            rpn = builder(cfg, in_channels, num_anchors)

            N, C_in, H, W = 2, in_channels, 24, 32
            input = torch.rand([N, C_in, H, W], dtype=torch.float32)
            LAYERS = 3
            out = rpn([input] * LAYERS)
            self.assertEqual(len(out), 2)
            logits, bbox_reg = out
            for idx in range(LAYERS):
                self.assertEqual(
                    logits[idx].shape,
                    torch.Size([
                        input.shape[0], num_anchors,
                        input.shape[2], input.shape[3],
                    ])
                )
                self.assertEqual(
                    bbox_reg[idx].shape,
                    torch.Size([
                        logits[idx].shape[0], num_anchors * 4,
                        logits[idx].shape[2], logits[idx].shape[3],
                    ]),
                )


if __name__ == "__main__":
    unittest.main()
