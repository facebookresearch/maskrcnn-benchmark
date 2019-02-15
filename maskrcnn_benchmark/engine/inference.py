# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import tempfile
import time
import os
import json
from collections import OrderedDict

import torch

from tqdm import tqdm

from ..utils.comm import is_main_process
from ..utils.comm import scatter_gather
from ..utils.comm import synchronize


def compute_on_dataset(model, data_loader, device):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for i, batch in tqdm(enumerate(data_loader)):
        images, targets, image_ids = batch
        images = images.to(device)
        with torch.no_grad():
            output = model(images)
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = scatter_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def save_as_bdd_format(preds, path, name, img_names):
    preds_bdd = []        
    for j in range(len(preds)):
        pred = preds[j]
        pred_bdd = {
            'name': img_names[j],
            'labels': []
        }
        boxes = pred.bbox.numpy().tolist()
        labels = pred.get_field('labels').numpy().tolist()
        scores = pred.get_field('scores').numpy().tolist()
        for i in range(len(boxes)):
            pred_bdd['labels'] += [{
                'category': labels[i],
                'box2d': {
                    'x1': boxes[i][0],
                    'y1': boxes[i][1],
                    'x2': boxes[i][2],
                    'y2': boxes[i][3]
                },
                'score': scores[i]
            }]
        preds_bdd += [pred_bdd]
    path = os.path.join(path, '{}.json'.format(name))
    with open(path, 'w') as f:
        json.dump(preds_bdd, f)


def inference(
    model,
    data_loader,
    iou_types=("bbox",),
    box_only=False,
    device="cuda",
    expected_results=(),
    expected_results_sigma_tol=4,
    output_folder=None,
    name="predictions"
):

    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = (
        torch.distributed.deprecated.get_world_size()
        if torch.distributed.deprecated.is_initialized()
        else 1
    )
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} images".format(len(dataset)))
    start_time = time.time()
    predictions = compute_on_dataset(model, data_loader, device)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Total inference time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        det_path = os.path.join(output_folder, "detections")
        if not os.path.exists(det_path):
            os.makedirs(det_path)
        save_as_bdd_format(predictions, det_path, name, dataset.image_paths)
    
    return
