# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch, os
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F

from maskrcnn_benchmark.structures.bounding_box import BoxList


CLASS_TYPE_CONVERSION = {
  'Pedestrian':     'person',
  'Cyclist':        'person',
  'Person_sitting': 'person',
  'Car':            'vehicle',
  'Van':            'vehicle',
  'Truck':          'vehicle'
}

TYPE_ID_CONVERSION = {
    'person': 0,
    'vehicle': 1
}

KITTI_MAX_WIDTH = 1242
KITTI_MAX_HEIGHT = 376

class KittiDataset(Dataset):
    """ KITTI Dataset: http://www.cvlibs.net/datasets/kitti/
    
  This Dataset implementation gets ROIFlow, which is just crops of valid
    detections compared with crops from adjacent anchor locations in adjacent
    frames, given a class value of the IoU with the anchor and the true track
    movement.
    """
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(Dataset, self).__init__()

        # TODO: sort indices for reproducible results

        # TODO: filter images without detection annotations
        
        self.transforms = transforms
        self.image_dir = os.path.join(root, 'images')
        self.label_dir = os.path.join(root, 'labels')
        image_ids = [d[:-4] for d in image_dir if d.endswith('.png')]
        label_ids = [d[:-4] for d in image_dir if d.endswith('.txt')]
        assert image_paths == label_ids
        self.length = len(image_ids)
        self.image_paths = [i + '.png' for i in image_ids]
        self.label_paths = [i + '.txt' for i in label_ids]
        
        
    def __len__(self):
        return self.length;
        

    def __getitem__(self, idx):
        
        # load image
        img = Image.open(self.image_paths[idx])
        
        # padding
        padBottom = KITTI_MAX_HEIGHT - img.size(1)
        padRight = KITTI_MAX_WIDTH - img.size(2)
        # (padLeft, padRight, padTop, padBottom)
        img = F.pad(0, padRight, 0, padBottom)
        
        # load annotations
        with open(self.label_paths[idx]) as f:
            labels = f.read().splitlines()
        
        boxes = []
        classes = []
        for label in labels:
            attributes = label.split(' ')
            if attributes[0] in CLASS_TYPE_CONVERSION.keys():
                # TODO: further filter annotations if needed
                
                label_type = CLASS_TYPE_CONVERSION[attributes[0]]
                classes += [TYPE_ID_CONVERSION[label_type]]
                boxes += [float(c) for c in attributes[4:8]]
        
        boxes = torch.as_tensor(boxes).reshape(-1, 4)
        target = BoxList(boxes, img.size, mode="xyxy")

        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
