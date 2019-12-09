# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints
import pandas as pd
from PIL import Image
import itertools
import os
import json
min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class InstanceSegDataset:
    def __init__(
        self,hparams, is_train,transforms=None
    ):
        self.hparams=hparams
        self.is_train=is_train
        self.hparams['input_dir']='/input0/'
        self.hparams['classlabels']=(
        "__background__ ",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )
        self.hparams['num_classes']=21
        self.root=self.hparams['input_dir']
        self.class_names=self.hparams['classlabels']
        assert len(self.class_names)==self.hparams['num_classes']
        self.classmap=dict(zip(self.hparams['classlabels'],range(len(self.hparams['classlabels']))))
        self.categories=dict(zip(range(len(self.class_names)),self.class_names))
        self._transforms=transforms
        if self.is_train:
            with open(os.path.join(self.root,'train_meta.csv')) as fr:
                self.ids=pd.read_csv(fr).values
        else:
            with open(os.path.join(self.root,'val_meta.csv')) as fr:
                self.ids=pd.read_csv(fr).values
        

    def __getitem__(self, idx):
        img_idx,anno_idx=self.ids[idx][1],self.ids[idx][0]
        img = Image.open(os.path.join(self.root, img_idx)).convert('RGB')
        with open(os.path.join(self.root,anno_idx)) as fr:
            anno=json.load(fr)
        image_height=anno['image_height']
        image_width=anno['image_width']
        boxes=[]
        classes=[]
        masks=[]
        for each in anno['bboxes']:
            x_min=each['x_min']*image_width
            x_max=each['x_max']*image_width
            y_min=each['y_min']*image_height
            y_max=each['y_max']*image_height
            mask=each['mask']
            if len(mask)==0:
                continue
            classes.append(each['label'])
            boxes.append([x_min,y_min,x_max-x_min,y_max-y_min])
            mask=[ [i,v] for i,v in zip(mask['all_points_x'],mask['all_points_y'])]
            mask=list(itertools.chain.from_iterable(mask))
            masks.append([mask])
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")
        classes=[self.classmap[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)
        #masks
        masks = SegmentationMask(masks, img.size, mode='poly')
        target.add_field("masks", masks)
        target = target.clip_to_image(remove_empty=True)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target, idx

    def get_img_info(self, index):
        print('*'*10,index)
        anno_idx=self.ids[index][0]
        with open(os.path.join(self.root,anno_idx)) as fr:
            anno=json.load(fr)
        image_height=anno['image_height']
        image_width=anno['image_width']
        img_data=dict(height=image_height,width=image_width)
        return img_data
    
    def __len__(self):
        return len(self.ids)
