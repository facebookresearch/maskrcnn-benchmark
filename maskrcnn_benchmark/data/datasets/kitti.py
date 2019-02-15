# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch, os
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import ToTensor

from maskrcnn_benchmark.structures.bounding_box import BoxList

# CLASS_TYPE_CONVERSION = {
#   'Pedestrian':     'pedestrian',
#   'Cyclist':        'cyclist',
#   'Person_sitting': 'person_sitting',
#   'Car':            'car',
#   'Van':            'van',
#   'Truck':          'truck'
# }

# CLASS_TYPE_CONVERSION = {
#   'Pedestrian':     'person',
#   'Cyclist':        'person',
#   'Person_sitting': 'person',
#   'Car':            'vehicle',
#   'Van':            'vehicle',
#   'Truck':          'vehicle'
# }

# TYPE_ID_CONVERSION = {
#     'pedestrian': 1,
#     'cyclist': 2,
#     'person_sitting': 3,
#     'car': 4,
#     'van': 5,
#     'truck': 6
# }

CLASS_TYPE_CONVERSION = {
  'Pedestrian':     'person',
  'Car':            'vehicle'
}

TYPE_ID_CONVERSION = {
    'person': 1,
    'vehicle': 2
}

KITTI_MAX_WIDTH = 1242
KITTI_MAX_HEIGHT = 376

class KittiDataset(Dataset):
    """ KITTI Dataset: http://www.cvlibs.net/datasets/kitti/
    
  This Dataset implementation gets- ROIFlow, which is just crops of valid
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
        self.image_paths = sorted([d for d in os.listdir(self.image_dir) if d.endswith('.png')])      
        self.label_paths = [d.replace('.png', '.txt') for d in self.image_paths]
#         assert len(self.image_paths) == len(self.label_paths)
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
        label_path = os.path.join(self.label_dir, self.label_paths[idx])
        target = None
        if os.path.exists(label_path):
            with open(label_path) as f:
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
        
    # Get all gt labels. Used in evaluation.
    def get_gt_labels(self):
        
        gt_labels = []
        
        for i, label_path in enumerate(self.label_paths):
            gt_label = {
                'name': os.path.join(self.image_dir, self.image_paths[i]),
                'labels': [],
            }
            with open(os.path.join(self.label_dir, label_path)) as f:
                labels = f.read().splitlines()
                
            for label in labels:
                attributes = label.split(' ')
                if attributes[0] in CLASS_TYPE_CONVERSION.keys():
                    # TODO: further filter annotations if needed

                    label_type = CLASS_TYPE_CONVERSION[attributes[0]]
                    category = TYPE_ID_CONVERSION[label_type]
                    box = [float(c) for c in attributes[4:8]]
                    
                    gt_label['labels'] += [{
                        'category': category,
                        'box2d': {
                            'x1': box[0],
                            'y1': box[1],
                            'x2': box[2],
                            'y2': box[3]
                        }
                    }]
                    
            gt_labels += [gt_label]
        
        return gt_labels


    def get_classes_ids(self):
        return TYPE_ID_CONVERSION;
    
