#!/usr/bin/env bash
python tools/train_net.py --config-file "configs/potato_mask_rcnn_R_50_FPN_1x.yaml" SOLVER.IMS_PER_BATCH 2
