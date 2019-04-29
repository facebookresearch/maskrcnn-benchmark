# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import os

import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
# from torch.nn.utils import clip_grad_value_

from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.utils.comm import get_world_size,synchronize
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.utils.comm import is_main_process

from tensorboardX import SummaryWriter

best_val_map = 0.0
cur_val_map = 0.0
is_best_val_map = False
writer = None

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    cfg,
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    distributed,
):
    global is_best_val_map, best_val_map, cur_val_map, writer

    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    if is_main_process():
        writer = SummaryWriter()

    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        total_norm = clip_grad_norm_(model.parameters(), cfg.SOLVER.GRAD_CLIP)
        # print('Total Norm: ', total_norm)
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        if is_main_process():
            writer.add_scalar('total_norm',total_norm, iteration)

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
            if is_main_process():
                writer.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], iteration)
                writer.add_scalar('train_loss', losses_reduced, iteration)
                for k,v in loss_dict_reduced.items():
                    writer.add_scalar(k, v.item(), iteration)

            # logger.info("Best Val mAP: {:.4f}".format(best_val_map))

        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)

        if iteration % cfg.SOLVER.VAL_PERIOD == 0:
            run_test(cfg, model, distributed, str(iteration))
            if is_best_val_map:
                logger.info("Best Val mAP: {:.4f}, Saving Best Model..".format(best_val_map))
                checkpointer.save("model_best", ** arguments)
            model.train()
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)
            run_test(cfg, model, distributed, 'final')

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
    if is_main_process():
        writer.export_scalars_to_json("./all_scalars.json")
        writer.close()
    logger.info("Best Val mAP: {:.4f}".format(best_val_map))

def run_test(cfg, model, distributed, iteration_name):
    global best_val_map, is_best_val_map, cur_val_map, writer
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name + '_' + iteration_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        results = inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        if not is_main_process():
            synchronize()
            return
        print('Comaptible matrix:', model.state_dict()['roi_heads.box.compatible_matrix'].cpu().data.numpy())
        print('Local Comaptible matrix:', model.state_dict()['roi_heads.box.local_compatible_matrix'].cpu().data.numpy())
        # print(model.state_dict().keys())
        if iteration_name != 'final':
            # # for coco evaluation
            if(dataset_name.startswith('coco')):
                for k,v in results.results.items():
                    for ki, vi in v.items():
                        if ki == 'AP':
                            cur_val_map = vi
                            if vi > best_val_map:
                                best_val_map = vi
                                is_best_val_map = True
                            else:
                                is_best_val_map = False
                        writer.add_scalar(dataset_name + '_' + k + '_' + ki, vi, int(iteration_name))
                        print(dataset_name + '_' + k + '_' + ki, vi)
            elif(dataset_name.startswith('voc')):
                # for VOC evaluation
                cur_val_map = results['map']
                if cur_val_map > best_val_map:
                    best_val_map = cur_val_map
                    is_best_val_map = True
                else:
                    is_best_val_map = False
                writer.add_scalar(dataset_name + '_map', cur_val_map, int(iteration_name))
        synchronize()
