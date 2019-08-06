# Abstract dataset definition for custom datasets
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
        super(AbstractCOCO, self).__init__()
        self.coco = COCODataset(
            ann_file, root, remove_images_without_annotations, transforms
        )
        self.classid_to_name = {
            key: value["name"] for key, value in self.coco.coco.cats.items()
        }
        self.initMaps()
        self.transforms = transforms

    def __getitem__(self, idx):
        return self.coco[idx]

    def get_img_info(self, index):
        return self.coco.get_img_info(index)

    def __len__(self):
        return len(self.coco)
