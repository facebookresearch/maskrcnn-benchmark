# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .abstract import AbstractDataset
from .coco import COCODataset
from .voc import PascalVOCDataset
from .concat_dataset import ConcatDataset

# Wrap an abstract dataset around COCODataset for testing if results are the same
# if computed using the generic COCO evaluation
# TODO: remove before merge
from .abstract_coco import AbstractCOCO

__all__ = ["AbstractDataset", "COCODataset", "ConcatDataset", "PascalVOCDataset", "AbstractCOCO"]
