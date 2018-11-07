
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import bisect

import torch.utils.data

from maskrcnn_benchmark.config.utils import import_file
from maskrcnn_benchmark.datasets.coco import COCODataset
from maskrcnn_benchmark.utils import data_transforms as T
from maskrcnn_benchmark.utils.concat_dataset import ConcatDataset
from maskrcnn_benchmark.utils.data_collate import BatchCollator
from maskrcnn_benchmark.utils.data_samplers import GroupedBatchSampler
from maskrcnn_benchmark.utils.data_samplers import compute_aspect_ratios
from maskrcnn_benchmark.utils.mlperf_logger import print_mlperf
from mlperf_compliance import mlperf_log


def make_transform(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_prob = 0

    resize_transform = T.Resize(min_size, max_size)

    to_bgr255 = True  # TODO make this an option?
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )

    transform = T.Compose(
        [
            resize_transform,
            T.RandomHorizontalFlip(flip_prob),
            T.ToTensor(),
            normalize_transform,
        ]
    )
    return transform


def make_coco_dataset(cfg, is_train=True):
    paths_catalog = import_file("maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True)
    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST

    transforms = make_transform(cfg, is_train)

    datasets = []
    for dataset_name in dataset_list:
        annotation_path, folder = DatasetCatalog.get(dataset_name)
        dataset = COCODataset(
            annotation_path,
            folder,
            remove_images_without_annotations=is_train,
            transforms=transforms,
        )
        datasets.append(dataset)

    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)

    return dataset


def make_data_sampler(dataset, shuffle, distributed):
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        assert (
            distributed == False
        ), "Distributed with no shuffling on the dataset not supported"
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def _quantize(x, bins):
    bins = sorted(bins.copy())
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def make_batch_data_sampler(dataset, sampler, aspect_grouping, images_per_batch):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = GroupedBatchSampler(
            sampler, group_ids, images_per_batch, drop_uneven=False
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=False
        )
    return batch_sampler


def make_data_loader(cfg, is_train=True, is_distributed=False):
    if is_train:
        images_per_batch = cfg.DATALOADER.IMAGES_PER_BATCH_TRAIN
        print_mlperf(key=mlperf_log.INPUT_ORDER)
        shuffle = True
    else:
        images_per_batch = cfg.DATALOADER.IMAGES_PER_BATCH_TEST
        shuffle = False if not is_distributed else True

    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []

    dataset = make_coco_dataset(cfg, is_train)
    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    batch_sampler = make_batch_data_sampler(
        dataset, sampler, aspect_grouping, images_per_batch
    )
    collator = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=collator,
        pin_memory=True
    )
    return data_loader
