# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import unittest
import copy
import torch
# import modules to to register predictors
from maskrcnn_benchmark.modeling.backbone import build_backbone # NoQA
from maskrcnn_benchmark.modeling.roi_heads.roi_heads import build_roi_heads # NoQA
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.config import cfg as g_cfg
from utils import load_config


# overwrite configs if specified, otherwise default config is used
PREDICTOR_CFGS = {
}

# overwrite configs if specified, otherwise default config is used
PREDICTOR_INPUT_CHANNELS = {
}


def _test_predictors(
    self, predictors, overwrite_cfgs, overwrite_in_channels,
    hwsize,
):
    ''' Make sure predictors run '''

    self.assertGreater(len(predictors), 0)

    in_channels_default = 64

    for name, builder in predictors.items():
        print('Testing {}...'.format(name))
        if name in overwrite_cfgs:
            cfg = load_config(overwrite_cfgs[name])
        else:
            # Use default config if config file is not specified
            cfg = copy.deepcopy(g_cfg)

        in_channels = overwrite_in_channels.get(
            name, in_channels_default)

        fe = builder(cfg, in_channels)

        N, C_in, H, W = 2, in_channels, hwsize, hwsize
        input = torch.rand([N, C_in, H, W], dtype=torch.float32)
        out = fe(input)
        yield input, out, cfg


class TestPredictors(unittest.TestCase):
    def test_roi_box_predictors(self):
        ''' Make sure roi box predictors run '''
        for cur_in, cur_out, cur_cfg in _test_predictors(
            self,
            registry.ROI_BOX_PREDICTOR,
            PREDICTOR_CFGS,
            PREDICTOR_INPUT_CHANNELS,
            hwsize=1,
        ):
            self.assertEqual(len(cur_out), 2)
            scores, bbox_deltas = cur_out[0], cur_out[1]
            self.assertEqual(
                scores.shape[1], cur_cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES)
            self.assertEqual(scores.shape[0], cur_in.shape[0])
            self.assertEqual(scores.shape[0], bbox_deltas.shape[0])
            self.assertEqual(scores.shape[1] * 4, bbox_deltas.shape[1])

    def test_roi_keypoints_predictors(self):
        ''' Make sure roi keypoint predictors run '''
        for cur_in, cur_out, cur_cfg in _test_predictors(
            self,
            registry.ROI_KEYPOINT_PREDICTOR,
            PREDICTOR_CFGS,
            PREDICTOR_INPUT_CHANNELS,
            hwsize=14,
        ):
            self.assertEqual(cur_out.shape[0], cur_in.shape[0])
            self.assertEqual(
                cur_out.shape[1], cur_cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES)

    def test_roi_mask_predictors(self):
        ''' Make sure roi mask predictors run '''
        for cur_in, cur_out, cur_cfg in _test_predictors(
            self,
            registry.ROI_MASK_PREDICTOR,
            PREDICTOR_CFGS,
            PREDICTOR_INPUT_CHANNELS,
            hwsize=14,
        ):
            self.assertEqual(cur_out.shape[0], cur_in.shape[0])
            self.assertEqual(
                cur_out.shape[1], cur_cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES)


if __name__ == "__main__":
    unittest.main()
