# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

r"""
Basic training script for PyTorch
"""
import argparse
import logging
import os

import torch
import random

from maskrcnn_benchmark.config import cfg


from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.engine.tester import test
from maskrcnn_benchmark.config.data import make_data_loader
from maskrcnn_benchmark.config.solver import make_optimizer
from maskrcnn_benchmark.config.solver import make_lr_scheduler
from maskrcnn_benchmark.config.utils import import_file
from maskrcnn_benchmark.modeling.model_builder import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import Checkpoint
from maskrcnn_benchmark.engine.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

import datetime
from mlperf_compliance import mlperf_log

from maskrcnn_benchmark.utils.mlperf_logger import print_mlperf
from maskrcnn_benchmark.utils.mlperf_logger import generate_seeds
from maskrcnn_benchmark.utils.mlperf_logger import broadcast_seeds

import time
import sys

# Known bug in cv2 - fixes hangs in dataloader
import cv2
cv2.setNumThreads(0)

try:
    from apex.parallel import DistributedDataParallel as DDP
    use_apex_ddp = True
except ImportError:
    print('Use APEX for better performance')
    use_apex_ddp = False

try:
    from apex import amp
    use_apex_amp = True
except ImportError:
    use_apex_amp = False


# TODO handle model retraining
def load_from_pretrained_checkpoint(cfg, model):
    if not cfg.MODEL.WEIGHT:
        return
    weight_path = cfg.MODEL.WEIGHT
    if weight_path.startswith("catalog://"):
        paths_catalog = import_file("maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True)
        ModelCatalog = paths_catalog.ModelCatalog
        weight_path = ModelCatalog.get(weight_path[len("catalog://"):])

    if weight_path.endswith("pkl"):
        from maskrcnn_benchmark.utils.c2_model_loading import _load_c2_pickled_weights, _rename_weights_for_R50
        state_dict = _load_c2_pickled_weights(weight_path)
        state_dict = _rename_weights_for_R50(state_dict)
    else:
        state_dict = torch.load(weight_path)
    if cfg.MODEL.RPN.USE_FPN or cfg.MODEL.ROI_HEADS.USE_FPN:
        model.backbone[0].stem.load_state_dict(state_dict, strict=False)
        model.backbone[0].load_state_dict(state_dict, strict=False)
    else:
        model.backbone.stem.load_state_dict(state_dict, strict=False)
        model.backbone.load_state_dict(state_dict, strict=False)

        model.roi_heads.heads[0].feature_extractor.head.load_state_dict(
            state_dict, strict=False
        )


def train(cfg, random_number_generator, local_rank, distributed, args, fp16=False):


    data_loader = make_data_loader(cfg, is_train=True, is_distributed=distributed)

    # todo sharath - undocument log below after package is updated
    # print_mlperf(key=mlperf_log.INPUT_SIZE, value=len(data_loader.dataset))

    print_mlperf(key=mlperf_log.INPUT_BATCH_SIZE, value=cfg.DATALOADER.IMAGES_PER_BATCH_TRAIN)
    print_mlperf(key=mlperf_log.BATCH_SIZE_TEST, value=cfg.DATALOADER.IMAGES_PER_BATCH_TEST)

    print_mlperf(key=mlperf_log.INPUT_MEAN_SUBTRACTION, value = cfg.INPUT.PIXEL_MEAN)
    print_mlperf(key=mlperf_log.INPUT_NORMALIZATION_STD, value=cfg.INPUT.PIXEL_STD)
    print_mlperf(key=mlperf_log.INPUT_RESIZE)
    print_mlperf(key=mlperf_log.INPUT_RESIZE_ASPECT_PRESERVING)
    print_mlperf(key=mlperf_log.MIN_IMAGE_SIZE, value=cfg.INPUT.MIN_SIZE_TRAIN)
    print_mlperf(key=mlperf_log.MAX_IMAGE_SIZE, value=cfg.INPUT.MAX_SIZE_TRAIN)
    print_mlperf(key=mlperf_log.INPUT_RANDOM_FLIP)
    print_mlperf(key=mlperf_log.RANDOM_FLIP_PROBABILITY, value=0.5)
    print_mlperf(key=mlperf_log.FG_IOU_THRESHOLD, value=cfg.MODEL.RPN.FG_IOU_THRESHOLD)
    print_mlperf(key=mlperf_log.BG_IOU_THRESHOLD, value=cfg.MODEL.RPN.BG_IOU_THRESHOLD)
    print_mlperf(key=mlperf_log.RPN_PRE_NMS_TOP_N_TRAIN, value=cfg.MODEL.RPN.PRE_NMS_TOP_N_TRAIN)
    print_mlperf(key=mlperf_log.RPN_PRE_NMS_TOP_N_TEST, value=cfg.MODEL.RPN.PRE_NMS_TOP_N_TEST)
    print_mlperf(key=mlperf_log.RPN_POST_NMS_TOP_N_TRAIN, value=cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN)
    print_mlperf(key=mlperf_log.RPN_POST_NMS_TOP_N_TEST, value=cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST)
    print_mlperf(key=mlperf_log.ASPECT_RATIOS, value=cfg.MODEL.RPN.ASPECT_RATIOS)

    print_mlperf(key=mlperf_log.BACKBONE, value=cfg.MODEL.BACKBONE.CONV_BODY)

    print_mlperf(key=mlperf_log.NMS_THRESHOLD, value=cfg.MODEL.RPN.NMS_THRESH)

    model = build_detection_model(cfg)
    load_from_pretrained_checkpoint(cfg, model)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    print_mlperf(key=mlperf_log.OPT_NAME, value=mlperf_log.SGD_WITH_MOMENTUM)
    print_mlperf(key=mlperf_log.OPT_LR, value=cfg.SOLVER.BASE_LR)
    print_mlperf(key=mlperf_log.OPT_MOMENTUM, value=cfg.SOLVER.MOMENTUM)
    print_mlperf(key=mlperf_log.OPT_WEIGHT_DECAY, value=cfg.SOLVER.WEIGHT_DECAY)

    scheduler = make_lr_scheduler(cfg, optimizer)
    max_iter = cfg.SOLVER.MAX_ITER

    if use_apex_amp:
        amp_handle = amp.init(enabled=fp16, verbose=False)
        if cfg.SOLVER.ACCUMULATE_GRAD:
            # also specify number of steps to accumulate over
            optimizer = amp_handle.wrap_optimizer(optimizer, num_loss=cfg.SOLVER.ACCUMULATE_STEPS)
        else:
            optimizer = amp_handle.wrap_optimizer(optimizer)

    if distributed:
        if use_apex_ddp:
            model = DDP(model, delay_allreduce=True)
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank
            )

    arguments = {}
    arguments["iteration"] = 0

    arguments["use_amp"] = use_apex_amp

    output_dir = cfg.OUTPUT_DIR

    if cfg.SAVE_CHECKPOINTS:
        checkpoint_file = cfg.CHECKPOINT
        checkpointer = Checkpoint(model, optimizer, scheduler, output_dir, local_rank)
        if checkpoint_file:
            extra_checkpoint_data = checkpointer.load(checkpoint_file)
            arguments.update(extra_checkpoint_data)
    else:
        checkpointer = None

    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        max_iter,
        device,
        distributed,
        arguments,
        cfg,
        args,
        random_number_generator,
    )

    return model

def main():

    mlperf_log.ROOT_DIR_MASKRCNN = os.path.dirname(os.path.abspath(__file__))
    # mlperf_log.LOGGER.propagate = False

    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch/configs/rpn_r50.py",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        '--skip-test',
        dest='skip_test',
        help='Do not test the model',
        action='store_true'
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable multi-precision training"
    )

    parser.add_argument(
        "--min_bbox_map",
        type=float,
        default=0.377,
        help="Target BBOX MAP"
    )

    parser.add_argument(
        "--min_mask_map",
        type=float,
        default=0.339,
        help="Target SEGM/MASK MAP"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed"
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    args.distributed = (
        int(os.environ["WORLD_SIZE"]) > 1 if "WORLD_SIZE" in os.environ else False
    )

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

        # to synchronize start of time
        torch.distributed.broadcast(torch.tensor([1], device="cuda"), 0)
        torch.cuda.synchronize()

        if torch.distributed.get_rank() == 0:
            # Setting logging file parameters for compliance logging
            os.environ["COMPLIANCE_FILE"] = '/MASKRCNN_complVv0.5.0_' + str(datetime.datetime.now())
            mlperf_log.LOG_FILE = os.getenv("COMPLIANCE_FILE")
            mlperf_log._FILE_HANDLER = logging.FileHandler(mlperf_log.LOG_FILE)
            mlperf_log._FILE_HANDLER.setLevel(logging.DEBUG)
            mlperf_log.LOGGER.addHandler(mlperf_log._FILE_HANDLER)

        print_mlperf(key=mlperf_log.RUN_START)

        # Setting seed
        seed_tensor = torch.tensor(0, dtype=torch.float32, device=torch.device("cuda"))

        if torch.distributed.get_rank() == 0:
            # seed = int(time.time())
            # random master seed, random.SystemRandom() uses /dev/urandom on Unix
            master_seed = random.SystemRandom().randint(0, 2 ** 32 - 1)

            seed_tensor = torch.tensor(master_seed, dtype=torch.float32, device=torch.device("cuda"))
        torch.distributed.broadcast(seed_tensor, 0)
        master_seed = int(seed_tensor.item())
    else:

        # Setting logging file parameters for compliance logging
        os.environ["COMPLIANCE_FILE"] = '/MASKRCNN_complVv0.5.0_' + str(datetime.datetime.now())
        mlperf_log.LOG_FILE = os.getenv("COMPLIANCE_FILE")
        mlperf_log._FILE_HANDLER = logging.FileHandler(mlperf_log.LOG_FILE)
        mlperf_log._FILE_HANDLER.setLevel(logging.DEBUG)
        mlperf_log.LOGGER.addHandler(mlperf_log._FILE_HANDLER)

        print_mlperf(key=mlperf_log.RUN_START)
        # random master seed, random.SystemRandom() uses /dev/urandom on Unix
        master_seed = random.SystemRandom().randint(0, 2 ** 32 - 1)

    args.seed = master_seed
    # random number generator with seed set to master_seed
    random_number_generator = random.Random(master_seed)
    print_mlperf(key=mlperf_log.RUN_SET_RANDOM_SEED, value=master_seed)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    if args.skip_test:
        cfg.DO_ONLINE_MAP_EVAL = False

    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    #logger = setup_logger("maskrcnn_benchmark", output_dir, args.local_rank)
    logger = setup_logger("maskrcnn_benchmark", None, args.local_rank)
    logger.info(args)

    # generate worker seeds, one seed for every distributed worker
    worker_seeds = generate_seeds(random_number_generator, torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1)

    # todo sharath what if CPU
    # broadcast seeds from rank=0 to other workers
    worker_seeds = broadcast_seeds(worker_seeds, device='cuda')

    # Setting worker seeds
    logger.info("Worker {}: Setting seed {}".format(args.local_rank, worker_seeds[args.local_rank]))
    torch.manual_seed(worker_seeds[args.local_rank])


    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = train(cfg, random_number_generator, args.local_rank, args.distributed, args, args.fp16)
    print_mlperf(key=mlperf_log.RUN_FINAL)
    # Since eval is done in trainer.py
    # if not args.skip_test:
    #     test(cfg, model, args.distributed)

if __name__ == "__main__":
    # Start clock
    now = time.time()
    main()
    print("&&&& MLPERF METRIC TIME=", time.time() - now)
