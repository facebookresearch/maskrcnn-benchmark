# Testing unit for PR #860
# https://github.com/facebookresearch/maskrcnn-benchmark/pull/860
# TODO: REMOVE THIS BEFORE MERGING
# by botcs@github

import torch
from .abstract import AbstractDataset
from .coco import COCODataset


class AbstractCOCO(AbstractDataset):
    def __init__(
        self,
        ann_file,
        root,
        remove_images_without_annotations=False,
        transforms=None,
    ):
        self.coco = COCODataset(
            ann_file, root, remove_images_without_annotations, transforms
        )

        # To make sure that the class IDs are contiguous and follows the same
        # order that was used during training.
        self.CLASSES = [(0, "__background__")] + [
            (cat["id"], cat["name"]) for cat in self.coco.coco.cats.values()
        ]
        self.CLASSES.sort(key=lambda x: x[0])
        self.CLASSES = [c[1] for c in self.CLASSES]
        self.initMaps()

        self.transforms = transforms

    def __getitem__(self, idx):
        return self.coco[idx]

    def get_img_info(self, index):
        return self.coco.get_img_info(index)

    def __len__(self):
        return len(self.coco)
