# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch, os, json
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import ToTensor

from maskrcnn_benchmark.structures.bounding_box import BoxList


CLASS_TYPE_CONVERSION = {
  'person':     'person',
  'rider':        'person',
  'car':            'vehicle',
  'bus':            'vehicle',
  'truck':          'vehicle'
}

TYPE_ID_CONVERSION = {
    'person': 0,
    'vehicle': 1
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
        self.image_paths = [d for d in os.listdir(self.image_dir) if d.endswith('.jpg')]
#         assert len(self.image_paths) == len(self.labels)
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

            label_type = CLASS_TYPE_CONVERSION[label['category']]
            classes += [TYPE_ID_CONVERSION[label_type]]
            boxes += [
                label['box2d']['x1'],
                label['box2d']['y1'],
                label['box2d']['x2'],
                label['box2d']['y2']
            ]

        boxes = torch.as_tensor(boxes).reshape(-1, 4)
        target = BoxList(boxes, (W, H), mode="xyxy")

        classes = torch.tensor(classes)
        target.add_field("labels", classes)
        return img, target, idx

    def get_img_info(self, idx):
        img = Image.open(os.path.join(self.image_dir, self.image_paths[idx]))
        return {'width': img.width, 'height': img.height}
