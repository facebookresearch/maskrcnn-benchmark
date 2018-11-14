# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch, os
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import ToTensor

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
        super(KittiDataset, self).__init__()

        # TODO: sort indices for reproducible results

        # TODO: filter images without detection annotations
        
        self.transforms = transforms
        self.image_dir = os.path.join(root, 'images')
        self.label_dir = os.path.join(root, 'labels')
        self.image_paths = [d for d in os.listdir(self.image_dir) if d.endswith('.png')]
        self.label_paths = [d for d in os.listdir(self.label_dir) if d.endswith('.txt')]
        assert len(self.image_paths) == len(self.label_paths)
        self.length = len(self.image_paths)
        
    def __len__(self):
        return self.length;
        

    def __getitem__(self, idx):
        
        # load image
        img = ToTensor()(Image.open(os.path.join(self.image_dir, self.image_paths[idx])))
        # padding
        padBottom = KITTI_MAX_HEIGHT - img.size(1)
        padRight = KITTI_MAX_WIDTH - img.size(2)
        # (padLeft, padRight, padTop, padBottom)
        img = F.pad(img, (0, padRight, 0, padBottom))
        
        # load annotations
        with open(os.path.join(self.label_dir, self.label_paths[idx])) as f:
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
        target = BoxList(boxes, (KITTI_MAX_WIDTH, KITTI_MAX_HEIGHT), mode="xyxy")

        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        return img, target, idx

    def get_img_info(self, idx):
        return {'width': KITTI_MAX_WIDTH, 'height': KITTI_MAX_HEIGHT}
