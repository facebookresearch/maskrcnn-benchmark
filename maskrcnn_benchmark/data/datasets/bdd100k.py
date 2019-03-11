# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch, os, json
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import ToTensor

from maskrcnn_benchmark.structures.bounding_box import BoxList


CLASS_TYPE_CONVERSION = {
  'person':     'person',
  'rider':      'rider',
  'car':        'car',
  'bus':        'bus',
  'truck':      'truck',
  'bike':       'bike',
  'motor':      'motor',
  'traffic light': 'traffic light',
  'traffic sign':  'traffic sign',
  'train':      'train'
}

TYPE_ID_CONVERSION = {
  'person':     1,
  'rider':      2,
  'car':        3,
  'bus':        4,
  'truck':      5,
  'bike':       6,
  'motor':      7,
  'traffic light': 8,
  'traffic sign':  9,
  'train':      10
}


class Bdd100kDataset(Dataset):
    """ BDD100k Dataset: https://bair.berkeley.edu/blog/2018/05/30/bdd/
    """
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(Bdd100kDataset, self).__init__()

        # TODO: filter images without detection annotations
        
        self.transforms = transforms
        self.image_dir = root
        with open(ann_file, 'r') as f:
            self.labels = json.load(f)
            
        # filter labels
        for i in range(len(self.labels)):
            self.labels[i]['labels'] = [l for l in self.labels[i]['labels']
                                        if l['category'] in CLASS_TYPE_CONVERSION.keys()]
        
        self.labels = [l for l in self.labels if len(l['labels']) > 0]
        for i in range(len(self.labels)):
            for j in range(len(self.labels[i]['labels'])):
                label_type = CLASS_TYPE_CONVERSION[self.labels[i]['labels'][j]['category']]
                self.labels[i]['labels'][j]['category'] = TYPE_ID_CONVERSION[label_type]
                
        self.image_paths = [os.path.join(self.image_dir, l['name']) for l in self.labels]
        self.length = len(self.labels)
        
    def __len__(self):
        return self.length;
        

    def __getitem__(self, idx):
        
        # annotations
        annotations = self.labels[idx]
        
        # load image
        img = Image.open(os.path.join(self.image_dir, annotations['name']))
        H, W = img.height, img.width
        img = ToTensor()(img)
        boxes = []
        classes = []
        for label in annotations['labels']:
            # TODO: further filter annotations if needed
            classes += [label['category']]
            boxes += [
                label['box2d']['x1'],
                label['box2d']['y1'],
                label['box2d']['x2'],
                label['box2d']['y2']
            ]
        fns = os.path.join(self.image_dir, annotations['name'])

        boxes = torch.as_tensor(boxes).reshape(-1, 4)
        target = BoxList(boxes, (W, H), mode="xyxy")

        classes = torch.tensor(classes)
        target.add_field("labels", classes)
        target.add_field('fn', fns)
        return img, target, idx

    def get_img_info(self, idx):
        return {'width': 1280, 'height': 720}
        
    # Get all gt labels. Used in evaluation.
    def get_gt_labels(self):
        
        return self.labels
        
    
    def get_classes_ids(self):
        return TYPE_ID_CONVERSION;
    
    
