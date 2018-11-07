# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Centralized catalog of paths."""

import os


class DatasetCatalog(object):
    DATA_DIR = "/coco"

    DATASETS = {
        "coco_2014_train": (
            "coco_train2014",
            "annotations/instances_train2014.json",
        ),
        "coco_2014_minival": (
            "coco_val2014",
            "annotations/instances_minival2014.json",
        ),
        "coco_2014_valminusminival": (
            "coco_val2014",
            "annotations/instances_valminusminival2014.json",
        ),
    }

    @staticmethod
    def get(name):
        return (
            os.path.join(DatasetCatalog.DATA_DIR, DatasetCatalog.DATASETS[name][1]),
            os.path.join(DatasetCatalog.DATA_DIR, DatasetCatalog.DATASETS[name][0]),
        )


class ModelCatalog(object):
    DATA_DIR = "/coco/models"
    MODELS = {"R-50": "R-50.pth"}

    @staticmethod
    def get(name):
        return os.path.join(ModelCatalog.DATA_DIR, ModelCatalog.MODELS[name])
