# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

import datetime
import logging
import time

import torch

from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.engine.tester import test
from maskrcnn_benchmark.utils.mlperf_logger import print_mlperf
from mlperf_compliance import mlperf_log
import random

def train_one_epoch(
    model, data_loader, optimizer, scheduler, device, iteration, max_iter, cfg,
    distributed, use_amp = False
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    meters = MetricLogger(delimiter="  ")
    model.train()
    end = time.time()

    # Optional gradient accumulation
    freq = cfg.SOLVER.ACCUMULATE_STEPS

    # Due to how the reference's custom batch sampler works, some processes may receive one more
    # local batch than the others.  This logic communicates which process is going to receive
    # the most local batches.
    #
    # We need to create the underlying iterator before we communicate, because the custom batch
    # sampler lazily avoids reorganizing the data for this epoch until iter() is actually called on it.
    data_loader_iter = iter(data_loader)
    num_batches = len(data_loader_iter)
    num_pad_batches = 0
    if distributed:
        num_batches_tensor = torch.cuda.LongTensor([num_batches])
        gathered = [torch.cuda.LongTensor(1) for i in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered, num_batches_tensor)
        gathered = [g.item() for g in gathered]
        num_pad_batches = max(gathered) - num_batches
        num_batches = max(gathered)

    # Processes that expect to receive fewer local batches pad the last iteration using
    # data duplicated from the beginning of the epoch.  Each epoch does treat every data element in
    # the dataset, some are just treated twice.
    #
    # The logic below allows the possibility that in a given epoch, the number of local batches
    # treated by each process may differ by more than one across processes.
    # In practice, I haven't observed them differing by more than one.
    #
    # TODO Does this need a compliance tag?  "Some tags are required once per run...and some are only
    # required if optional code is included such as padding."
    # https://github.com/mlperf/policies/blob/master/training_rules.adoc#151-submission-compliance-logs
    def padded_prefetcher(load_iterator, num_pad_batches):
        prefetch_stream = torch.cuda.Stream()
        pad_batches = []

        def _prefetch():
            try:
                # I'm not sure why the trailing _ is necessary but the reference used
                # "for i, (images, targets, _) in enumerate(data_loader):" so I'll keep it.
                images, targets, _ = next(load_iterator)
            except StopIteration:
                return None, None

            with torch.cuda.stream(prefetch_stream):
                # TODO:  I'm not sure if the dataloader knows how to pin the targets' datatype.
                targets = [target.to(device, non_blocking=True) for target in targets]
                images = images.to(device, non_blocking=True)

            return images, targets

        next_images, next_targets = _prefetch()

        while next_images is not None:
            torch.cuda.current_stream().wait_stream(prefetch_stream)
            current_images, current_targets = next_images, next_targets
            next_images, next_targets = _prefetch()
            if len(pad_batches) < num_pad_batches:
                pad_batches.append((current_images, current_targets))
            yield current_images, current_targets

        for current_images, current_targets in pad_batches:
            yield current_images, current_targets


    for p in model.parameters():
        p.grad = None

    for i, (images, targets) in enumerate(padded_prefetcher(data_loader_iter, num_pad_batches)):
        data_time = time.time() - end

        scheduler.step()

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        meters.update(loss=losses, **loss_dict)

        if use_amp:
            with optimizer.scale_loss(losses) as scaled_losses:
                scaled_losses.backward()
        else:
            losses.backward()

        #accumulate gradient every 'freq' iters if set
        if not cfg.SOLVER.ACCUMULATE_GRAD:
            optimizer.step()
            # There should be a nice CPU-GPU skew here, where this overhead can fit benignly.
            for p in model.parameters():
                p.grad = None
        else:
            if (iteration + 1) % freq == 0:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.data.div_(freq)

                optimizer.step()
                for p in model.parameters():
                    p.grad = None

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == (max_iter - 1):
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

        iteration += 1
        if iteration >= max_iter:
            break
    return iteration

def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    max_iter,
    device,
    use_distributed,
    arguments,
    config,
    args,
    random_number_generator,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    start_training_time = time.time()

    print_mlperf(key=mlperf_log.TRAIN_LOOP)

    epoch = 0
    while arguments["iteration"] < max_iter:

        print_mlperf(key=mlperf_log.TRAIN_EPOCH, value=epoch)
        start_epoch_time = time.time()
        iteration = arguments["iteration"]
        if use_distributed:

            # Using Random number generator with master seed to generate random seeds for every epoch
            iteration_seed = random_number_generator.randint(0, 2**32 - 1)
            data_loader.batch_sampler.sampler.set_epoch(iteration_seed)

        iteration_end = train_one_epoch(
            model, data_loader, optimizer, scheduler, device, iteration, max_iter, config,
            use_distributed, use_amp=arguments["use_amp"]
        )
        total_epoch_time = time.time() - start_epoch_time

        epoch_time_str = str(datetime.timedelta(seconds=total_epoch_time))
        logger.info(
            "Total epoch time: {} ({:.4f} s / it)".format(
                epoch_time_str, total_epoch_time / (iteration_end - iteration)
            )
        )
        arguments["iteration"] = iteration_end

        if checkpointer:
            checkpointer("model_{}".format(arguments["iteration"]), **arguments)


        if config.DO_ONLINE_MAP_EVAL:

            print_mlperf(key=mlperf_log.EVAL_START, value=epoch)
            results = test(config, model, use_distributed)

            print_mlperf(key=mlperf_log.EVAL_TARGET, value={"BBOX": 0.377,
                                                              "SEGM": 0.339})
            map_tensor = torch.tensor((0, 0), dtype=torch.float32, device=torch.device("cuda"))

            if results: #Rank 0 process
                bbox_map = results['bbox']
                mask_map = results['segm']
                map_tensor = torch.tensor((bbox_map, mask_map), dtype=torch.float32, device=torch.device("cuda"))

            if use_distributed:
                torch.distributed.broadcast(map_tensor, 0)
                bbox_map = map_tensor[0].item()
                mask_map = map_tensor[1].item()


            logger.info("bbox map: {} mask map: {}".format(bbox_map, mask_map))
            print_mlperf(key=mlperf_log.EVAL_ACCURACY, value={"epoch":epoch, "value":{"BBOX":bbox_map, "SEGM":mask_map}})
            print_mlperf(key=mlperf_log.EVAL_STOP)

            # Terminating condition
            if bbox_map >= args.min_bbox_map and mask_map >= args.min_mask_map:
                logger.info("Target MAP reached. Exiting...")
                print_mlperf(key=mlperf_log.RUN_STOP, value={"success":True})
                break
        epoch += 1

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (arguments["iteration"])
        )
    )
    logger.info(
        "&&&& MLPERF METRIC THROUGHPUT per GPU={:.4f} iterations / s".format((arguments["iteration"] * 1.0) / total_training_time)
        )
