# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

import argparse
import os

import torch

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.config.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.logger import setup_logger
from maskrcnn_benchmark.modeling.model_builder import build_detection_model
from maskrcnn_benchmark.utils.c2_model_loading import load_from_c2
from maskrcnn_benchmark.modeling.utils import load_state_dict


def load_from_checkpoint(cfg, model, checkpoint):
    if cfg.MODEL.C2_COMPAT.ENABLED:
        load_from_c2(cfg, model, checkpoint)
        return

    # do standard loading here
    checkpoint = torch.load(checkpoint)
    # TODO find a better way of serializing the weights
    # that avoids this ugly workaround
    model = torch.nn.DataParallel(model)
    load_state_dict(model, checkpoint["model"])
    model = model.module


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--checkpoint", default="", metavar="FILE", help="path to checkpoint file"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    distributed = (
        int(os.environ["WORLD_SIZE"]) > 1 if "WORLD_SIZE" in os.environ else False
    )

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, args.local_rank)
    logger.info(cfg)

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    load_from_checkpoint(cfg, model, args.checkpoint)

    data_loader_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    inference(model, data_loader_val, iou_types=iou_types, box_only=cfg.MODEL.RPN_ONLY, device=cfg.MODEL.DEVICE)


if __name__ == "__main__":
    main()
