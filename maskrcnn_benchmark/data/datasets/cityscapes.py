import os
import glob
import json
from PIL import Image


import numpy as np
import torch
import torchvision


from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from .abstract import AbstractDataset

from cityscapesscripts.helpers import csHelpers


class CityScapesDataset(AbstractDataset):
    def __init__(
        self,
        img_dir,
        ann_dir,
        split,
        mode="mask",
        transforms=None,
        min_area=0,
        mini=None,
    ):
        """
        Arguments:
            img_dir: /path/to/leftImg8bit/      has to contain {train,val,test}
            ann_dir: /path/to/gtFine/           has to contain {train,val,test}
            split: "train" or "val" or "test"
            mode: "poly" or "mask", which annotation format to use
            transforms: apply transformations to input/annotation
            min_area: exclude intances below a specific area (bbox area)
            mini: limit the size of the dataset, so len(dataset) == mini for
                debugging purposes
        """
        assert split in ["train", "val", "test"]

        img_dir = os.path.abspath(os.path.join(img_dir, split))
        ann_dir = os.path.abspath(os.path.join(ann_dir, split))

        assert os.path.exists(img_dir), img_dir
        assert os.path.exists(ann_dir), ann_dr

        self.ann_dir = ann_dir

        self.split = split
        self.CLASSES = ["__background__"]
        self.CLASSES += [l.name for l in csHelpers.labels if l.hasInstances]

        # Adds name_to_id and id_to_name mapping
        self.initMaps()

        # This is required for parsing binary masks
        self.cityscapesID_to_ind = {
            l.id: self.name_to_id[l.name] for l in csHelpers.labels if l.hasInstances
        }

        self.transforms = transforms
        self.min_area = int(min_area)

        img_pattern = os.path.join(img_dir, "*", "*_leftImg8bit.png")
        img_paths = sorted(glob.glob(img_pattern))

        if mode == "mask":
            ann_pattern = os.path.join(ann_dir, "*", "*_instanceIds.png")
        elif mode == "poly":
            ann_pattern = os.path.join(ann_dir, "*", "*_polygons.json")
        else:
            raise NotImplementedError("Mode is not implemented yet: %s" % mode)

        self.mode = mode
        ann_paths = sorted(glob.glob(ann_pattern))

        if mini is not None:
            # Keep the mini dataset diverse by setting the stride
            img_paths = img_paths[:: len(img_paths) // mini + 1]
            ann_paths = ann_paths[:: len(ann_paths) // mini + 1]

        assert len(img_paths) == len(ann_paths)

        self.img_paths = img_paths
        self.ann_paths = ann_paths

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        ann_path = self.ann_paths[idx]

        if self.mode == "mask":
            ann = torch.from_numpy(np.asarray(Image.open(ann_path)))
            # masks are represented with tensors
            boxes, segmentations, labels = self._processBinayMasks(ann)
        else:
            with open(ann_path, "r") as ann_file:
                ann = json.load(ann_file)
            # masks are represented with polygons
            boxes, segmentations, labels = self._processPolygons(ann)

        boxes, segmentations, labels = self._filterGT(boxes, segmentations, labels)

        if len(segmentations) == 0:
            empty_ann_path = self.get_img_info(idx)["ann_path"]
            print("EMPTY ENTRY:", empty_ann_path)
            # self.img_paths.pop(idx)
            # self.ann_paths.pop(idx)
            img, target, _ = self[(idx + 1) % len(self)]

            # just override this image with the next
            return img, target, idx

        img = Image.open(img_path)
        # Compose all into a BoxList instance
        target = BoxList(boxes, img.size, mode="xyxy")
        target.add_field("labels", torch.tensor(labels))
        masks = SegmentationMask(segmentations, img.size, mode=self.mode)
        target.add_field("masks", masks)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx

    def _filterGT(self, boxes, segmentations, labels):
        filtered_boxes = []
        filtered_segmentations = []
        filtered_labels = []
        assert len(segmentations) == len(labels) == len(boxes)

        for box, segmentation, label in zip(boxes, segmentations, labels):
            xmin, ymin, xmax, ymax = box
            area = (xmax - xmin) * (ymax - ymin)
            if area < self.min_area:
                continue

            filtered_boxes.append(box)
            filtered_segmentations.append(segmentation)
            filtered_labels.append(label)

        if len(filtered_boxes) < 1:
            filtered_boxes = torch.empty(0, 4)

        return filtered_boxes, filtered_segmentations, filtered_labels

    def _processPolygons(self, ann):
        # For a single object polygon annotations are stored in CityScapes like
        # [[x1, y1], [x2, y2]...] and we need them in the following format:
        # [x1, y1, x2, y2, x3, y3 ...]
        polys = []
        labels = []
        boxes = []

        def poly_to_tight_box(poly):
            xmin = int(min(poly[::2]))
            ymin = int(min(poly[1::2]))
            xmax = int(max(poly[::2]))
            ymax = int(max(poly[1::2]))
            bbox = xmin, ymin, xmax, ymax
            return bbox

        for inst in ann["objects"]:
            label = inst["label"]
            if label not in self.CLASSES:
                continue

            label = self.name_to_id[label]

            cityscapes_poly = inst["polygon"]
            poly = []
            for xy in cityscapes_poly:
                # Equivalent with `poly += xy` but this is more verbose
                x = xy[0]
                y = xy[1]
                poly.append(x)
                poly.append(y)

            # In CityScapes instances are described with single polygons only
            box = poly_to_tight_box(poly)

            boxes.append(box)
            polys.append([poly])
            labels.append(label)

        if len(boxes) < 1:
            boxes = torch.empty(0, 4)

        return boxes, polys, labels

    def _processBinayMasks(self, ann):
        boxes = []
        masks = []
        labels = []

        def mask_to_tight_box(mask):
            a = mask.nonzero()
            bbox = [
                torch.min(a[:, 1]),
                torch.min(a[:, 0]),
                torch.max(a[:, 1]),
                torch.max(a[:, 0]),
            ]
            bbox = list(map(int, bbox))
            return bbox  # xmin, ymin, xmax, ymax

        # Sort for consistent order between instances as the polygon annotation
        instIds = torch.sort(torch.unique(ann))[0]
        for instId in instIds:
            if instId < 1000:  # group labels
                continue

            mask = ann == instId
            label = int(instId / 1000)
            label = self.cityscapesID_to_ind[label]
            box = mask_to_tight_box(mask)

            boxes.append(box)
            masks.append(mask)
            labels.append(label)

        return boxes, masks, labels

    def __len__(self):
        return len(self.img_paths)

    def get_img_info(self, index):
        # Reverse engineered from voc.py
        # All the images have the same size
        return {
            "height": 1024,
            "width": 2048,
            "idx": index,
            "img_path": self.img_paths[index],
            "ann_path": self.ann_paths[index],
        }
