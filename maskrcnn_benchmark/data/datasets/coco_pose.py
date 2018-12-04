import torch

import os

import numpy as np

import pycocotools.mask as mask_utils

from .coco import COCODataset

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.vertex_mask import VertexMask


def _generate_vertex_center_mask(label_mask, center, z):
    # c = np.zeros((2, 1), dtype=np.float32)
    # for ind, cls in enumerate(cls_i):
    c = np.expand_dims(center, axis=1) 
    h,w = label_mask.shape
    vertex_centers = np.zeros((3,h,w),dtype=np.float32)  # channels first, as in pytorch convention
    # z = pose[2, 3]
    y, x = np.where(label_mask == 1)

    R = c - np.vstack((x, y))
    # compute the norm
    N = np.linalg.norm(R, axis=0) + 1e-10
    # normalization
    R = R / N # np.divide(R, np.tile(N, (2,1)))
    # assignment
    vertex_centers[0, y, x] = R[0,:]
    vertex_centers[1, y, x] = R[1,:]
    vertex_centers[2, y, x] = z
    return vertex_centers

def _get_mask_from_polygon(polygons, im_size):
    width, height = im_size
    rles = mask_utils.frPyObjects(polygons, height, width)
    rle = mask_utils.merge(rles)
    mask = mask_utils.decode(rle)
    return mask

class COCOPoseDataset(COCODataset):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(COCOPoseDataset, self).__init__(ann_file, root, remove_images_without_annotations, transforms)

        # self.num_classes = len(self.json_category_id_to_contiguous_id)
        categories = self.coco.cats
        self._classes = ['__background'] + [categories[k]['name'] for k in categories]  # +1 for bg class
        self.num_classes = len(self._classes)

        extents_file = os.path.join(root, "../extents.txt")
        symmetry_file = os.path.join(root, "../symmetry.txt")
        models_dir = os.path.join(root, "../models")

        # read symmetry file
        self.symmetry = self._load_object_symmetry(symmetry_file)
        self.symmetry = torch.tensor(self.symmetry)

        # read points from models_dir
        _, self.points = self._load_object_points(models_dir)
        self.points = torch.tensor(self.points)
        # maybe get 'extents' from points instead?

        # read extents file
        self.extents = self._load_object_extents(extents_file)
        self.extents = torch.tensor(self.extents)


    def _load_object_points(self, models_dir):

        points = [[] for _ in range(self.num_classes)]
        num = np.inf

        for i in range(1, self.num_classes):
            point_file = os.path.join(models_dir, self._classes[i], 'points.xyz')
            # print point_file
            assert os.path.exists(point_file), 'Path does not exist: {}'.format(point_file)
            points[i] = np.loadtxt(point_file)
            if points[i].shape[0] < num:
                num = points[i].shape[0]

        points_all = np.zeros((self.num_classes, num, 3), dtype=np.float32)
        for i in range(1, self.num_classes):
            points_all[i, :, :] = points[i][:num, :]

        return points, points_all


    def _load_object_extents(self, extent_file):

        assert os.path.exists(extent_file), \
                'Path does not exist: {}'.format(extent_file)

        extents = np.zeros((self.num_classes, 3), dtype=np.float32)
        extents[1:, :] = np.loadtxt(extent_file)

        return extents

    def _load_object_symmetry(self, symmetry_file):

        assert os.path.exists(symmetry_file), \
                'Path does not exist: {}'.format(symmetry_file)

        symmetry = np.zeros((self.num_classes), dtype=np.float32)
        symmetry[1:] = np.loadtxt(symmetry_file)

        return symmetry

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        # classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        polygons = [obj["segmentation"] for obj in anno]
        seg_mask_instance = SegmentationMask(polygons, img.size)
        target.add_field("masks", seg_mask_instance)

        meta = [obj["meta"] for obj in anno]
        poses = [obj["pose"] for obj in meta]

        target.add_field("poses", torch.tensor(poses))

        assert len(meta) == len(polygons)
        # masks = [_get_mask_from_polygon(polygon, img.size) for polygon in seg_mask_instance.polygons]
        vertex_centers = []
        for ix, poly in enumerate(polygons):
            center = meta[ix]['center']
            pose = poses[ix]
            z = np.log(pose[-1]) # z distance is the last value in pose [qw,qx,qy,qz,x,y,z]
            m = _get_mask_from_polygon(poly, img.size)
            vertex_centers.append(_generate_vertex_center_mask(m, center, z))
        vertex_centers = torch.tensor(vertex_centers)
        vertexes = VertexMask(vertex_centers, img.size)
        target.add_field("vertex", vertexes)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        target.add_field("symmetry", self.symmetry)
        target.add_field("extents", self.extents)
        target.add_field("points", self.points)

        return img, target, idx        
