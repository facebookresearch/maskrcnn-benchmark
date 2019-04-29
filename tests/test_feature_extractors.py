# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import unittest
import copy
import torch
# import modules to to register feature extractors
from maskrcnn_benchmark.modeling.backbone import build_backbone # NoQA
from maskrcnn_benchmark.modeling.roi_heads.roi_heads import build_roi_heads # NoQA
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.config import cfg as g_cfg
from utils import load_config

# overwrite configs if specified, otherwise default config is used
FEATURE_EXTRACTORS_CFGS = {
}

# overwrite configs if specified, otherwise default config is used
FEATURE_EXTRACTORS_INPUT_CHANNELS = {
    # in_channels was not used, load through config
    "ResNet50Conv5ROIFeatureExtractor": 1024,
}


def _test_feature_extractors(
    self, extractors, overwrite_cfgs, overwrite_in_channels
):
    ''' Make sure roi box feature extractors run '''

    self.assertGreater(len(extractors), 0)

    in_channels_default = 64

    for name, builder in extractors.items():
        print('Testing {}...'.format(name))
        if name in overwrite_cfgs:
            cfg = load_config(overwrite_cfgs[name])
        else:
            # Use default config if config file is not specified
            cfg = copy.deepcopy(g_cfg)

        in_channels = overwrite_in_channels.get(
            name, in_channels_default)

        fe = builder(cfg, in_channels)
        self.assertIsNotNone(
            getattr(fe, 'out_channels', None),
            'Need to provide out_channels for feature extractor {}'.format(name)
        )

        N, C_in, H, W = 2, in_channels, 24, 32
        input = torch.rand([N, C_in, H, W], dtype=torch.float32)
        bboxes = [[1, 1, 10, 10], [5, 5, 8, 8], [2, 2, 3, 4]]
        img_size = [384, 512]
        box_list = BoxList(bboxes, img_size, "xyxy")
        out = fe([input], [box_list] * N)
        self.assertEqual(
            out.shape[:2],
            torch.Size([N * len(bboxes), fe.out_channels])
        )


class TestFeatureExtractors(unittest.TestCase):
    def test_roi_box_feature_extractors(self):
        ''' Make sure roi box feature extractors run '''
        _test_feature_extractors(
            self,
            registry.ROI_BOX_FEATURE_EXTRACTORS,
            FEATURE_EXTRACTORS_CFGS,
            FEATURE_EXTRACTORS_INPUT_CHANNELS,
        )

    def test_roi_keypoints_feature_extractors(self):
        ''' Make sure roi keypoints feature extractors run '''
        _test_feature_extractors(
            self,
            registry.ROI_KEYPOINT_FEATURE_EXTRACTORS,
            FEATURE_EXTRACTORS_CFGS,
            FEATURE_EXTRACTORS_INPUT_CHANNELS,
        )

    def test_roi_mask_feature_extractors(self):
        ''' Make sure roi mask feature extractors run '''
        _test_feature_extractors(
            self,
            registry.ROI_MASK_FEATURE_EXTRACTORS,
            FEATURE_EXTRACTORS_CFGS,
            FEATURE_EXTRACTORS_INPUT_CHANNELS,
        )


if __name__ == "__main__":
    unittest.main()
