#!/usr/bin/env bash
python tools/test_net.py --config-file "configs/potato_mask_rcnn_R_50_FPN_1x.yaml" TEST.IMS_PER_BATCH 4
