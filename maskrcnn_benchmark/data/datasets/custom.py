'''Load image/labels/boxes from an annotation file.

The list file is like:

    img.jpg xmin ymin xmax ymax label xmin ymin xmax ymax label ...
'''
import torch
import torch.utils.data as data
from PIL import Image
import requests
from io import BytesIO

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask


class CustomDataset(data.Dataset):
    def __init__(self, annotations, transforms, classes=None):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str) path to index file.
          transform: ([transforms]) image transforms.
          input_size: (int) model input size.
        '''
        self.annotations = annotations
        self.transforms = transforms

        self.img_pathes = []
        self.boxes = []
        self.labels = []
        self.masks = []
        self._cached_images = {}
        self.num_samples = len(self.annotations)

        if classes:
            self.classes = classes
        else:
            _classes_set = set()
            for _item in self.annotations:
                for _object in _item['objects']:
                    _classes_set.add(_object['label'])
            self.classes = _classes_set
        self._relabel_object_classes()

        for item in self.annotations:
            self.img_pathes.append(item['img_path'])
            boxes = []
            masks = []
            labels = []
            for row in item['objects']:
                box = row['bbox']
                mask = row['polygon']
                label = self.class_to_ind[row['label']]
                boxes.append([item for sublist in box for item in sublist])
                masks.append([item for sublist in mask for item in sublist])
                labels.append(label)
            self.boxes.append(torch.Tensor(boxes))
            self.masks.append(masks)
            self.labels.append(torch.LongTensor(labels))

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_targets: (tensor) location targets.
          cls_targets: (tensor) class label targets.
        '''
        # Load image and boxes.
        img_path = self.img_pathes[idx]
        if img_path not in self._cached_images.keys():
            self._cached_images[img_path] = self.get_image(img_path)
        img = self._cached_images[img_path]

        boxes = self.boxes[idx].clone()
        masks = self.masks[idx]
        target = BoxList(boxes, img.size, mode="xyxy")
        labels = self.labels[idx]
        target.add_field("labels", labels)
        masks = [[m] for m in masks]
        masks = SegmentationMask(masks, img.size)
        target.add_field("masks", masks)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx

    def _relabel_object_classes(self):
        self.num_classes = len(self.classes)
        self.class_label = sorted(list(self.classes))
        self.class_to_ind = dict(zip(self.class_label, range(self.num_classes)))
        self.ind_to_class = dict(zip(range(self.num_classes), self.class_label))

    def get_image(self, img_path):
        response = requests.get(img_path)
        img = Image.open(BytesIO(response.content))
        return img

    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        img_path = self.img_pathes[idx]
        if img_path not in self._cached_images.keys():
            self._cached_images[img_path] = self.get_image(img_path)
        img = self._cached_images[img_path]
        return {"height": img.size[1], "width": img.size[1]}

    def __len__(self):
        return self.num_samples
