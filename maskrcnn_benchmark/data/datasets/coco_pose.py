import torch

import os

from PIL import Image

import numpy as np

import pycocotools.mask as mask_utils

from .coco import COCODataset

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask, Polygons
from maskrcnn_benchmark.structures.object_mask import ObjectMask


def _generate_vertex_center_mask(label_mask, center, depth=None):
    # c = np.zeros((2, 1), dtype=np.float32)
    # for ind, cls in enumerate(cls_i):
    c = np.expand_dims(center, axis=1) 
    h,w = label_mask.shape
    vertex_centers = np.zeros((3, h, w), dtype=np.float32)  # channels first, as in pytorch convention
    # z = pose[2, 3]
    y, x = np.where(label_mask == 1)

    R = c - np.vstack((x, y))
    # compute the norm
    N = np.linalg.norm(R, axis=0) + 1e-10
    # normalization
    R = R / N # np.divide(R, np.tile(N, (2,1)))
    # assignment
    vertex_centers[0, y, x] = R[0, :]
    vertex_centers[1, y, x] = R[1, :]
    if depth is not None:
        assert depth.shape == (h, w)
        vertex_centers[2, y, x] = depth[y, x]
    return vertex_centers

def _generate_depth_mask(label_mask, depth):
    h,w = label_mask.shape
    assert depth.shape == (h, w)

    depths = np.zeros((1, h, w), dtype=np.float32)  # channels first, as in pytorch convention
    y, x = np.where(label_mask == 1)
    depths[0, y, x] = depth[y, x]
    return depths

def _get_mask_from_polygon(polygons, im_size):
    width, height = im_size
    rles = mask_utils.frPyObjects(polygons, height, width)
    rle = mask_utils.merge(rles)
    mask = mask_utils.decode(rle)
    return mask

class COCOPoseDataset(COCODataset):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None, cfg=None
    ):
        super(COCOPoseDataset, self).__init__(ann_file, root, remove_images_without_annotations, transforms)

        # self.num_classes = len(self.json_category_id_to_contiguous_id)
        categories = self.coco.cats
        self._classes = ['__background'] + [categories[k]['name'] for k in categories]  # +1 for bg class
        self.num_classes = len(self._classes)

        self.cfg = {"Pose": False, "Vertex": False, "Depth": False}
        if cfg is not None:
            self.cfg["Vertex"] = cfg.MODEL.VERTEX_ON
            self.cfg["Depth"] = cfg.MODEL.DEPTH_ON
            self.cfg["Pose"] = cfg.MODEL.POSE_ON

        if self.cfg["Pose"]:
            # extents_file = os.path.join(root, "../extents.txt")
            models_dir = os.path.join(root, "../models")

            # read points from models_dir
            _, self.points = self._load_object_points(models_dir)
            # maybe get 'extents' from points instead?

            # read extents file
            # self.extents = self._load_object_extents(extents_file)
            # self.extents = np.zeros((len(self.points), 3))
            self.extents = np.max(self.points,axis=1) - np.min(self.points,axis=1)

            # set weights to points
            # read symmetry file
            symmetry_file = os.path.join(root, "../symmetry.txt")
            self.symmetry = self._load_object_symmetry(symmetry_file)
            is_symmetric = True
            point_blob = self.points.copy()
            for i in range(1, self.num_classes):
                # compute the rescaling factor for the points
                weight = 2.0 / np.amax(self.extents[i, :])
                weight = max(weight, 10.0)
                if self.symmetry[i] > 0: #and is_symmetric:
                    weight *= 4
                point_blob[i, :, :] *= weight

            self.symmetry = torch.tensor(self.symmetry)
            self.points = torch.tensor(point_blob)
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

    def __get_item__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        data = coco.loadImgs(img_id)[0]
        f_name = data['file_name']
        img_path = os.path.join(self.root, f_name)
        img = Image.open(img_path).convert('RGB')


        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        depth = None
        if self.cfg["Depth"]:
            if 'depth_file_name' in data:
                depth_path = os.path.join(self.root, data['depth_file_name'])
                depth = np.array(Image.open(depth_path)).astype(np.float32)
                if 'factor_depth' in data:
                    depth = depth / float(data['factor_depth'])
            else:
                print("[WARN]: Depth mode is ON, but no 'depth_file_name' field in annotation: %s"%(f_name))

        return img, target, depth

    def __getitem__(self, idx):
        img, anno, depth = self.__get_item__(idx)

        # filter crowd annotations

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

        masks = [_get_mask_from_polygon(polygon, img.size) for polygon in polygons]
        N = len(masks)
        W, H = img.size
        if self.cfg["Pose"] or self.cfg["Vertex"]:
            meta = [obj["meta"] for obj in anno]
            centers = [m['center'] for m in meta]
            assert len(meta) == len(polygons)

            if self.cfg["Pose"]:
                poses = [obj["pose"] for obj in meta]
                target.add_field("poses", torch.tensor(poses))

            if self.cfg["Vertex"]:
                vertex_centers = np.zeros((N, 3, H, W))
                for ix, m in enumerate(masks):
                    center = centers[ix]
                    # pose = poses[ix]
                    # z = np.log(pose[-1]) # z distance is the last value in pose [qw,qx,qy,qz,x,y,z]
                    # m = _get_mask_from_polygon(poly, img.size)
                    vertex_centers[ix, :] = _generate_vertex_center_mask(m, center, depth)

                vertex_centers = torch.tensor(vertex_centers)
                vertexes = ObjectMask(vertex_centers, img.size)
                target.add_field("vertex", vertexes)

            centers = Polygons(centers, img.size, mode=None)
            target.add_field("centers", centers)

        if self.cfg["Depth"]:
            depth_data = np.zeros((N, 1, H, W))
            if depth is not None:
                for ix, m in enumerate(masks):
                    depth_data[ix, :] = _generate_depth_mask(m, depth)
            depth_data = torch.tensor(depth_data)
            depth_D = ObjectMask(depth_data, img.size)
            target.add_field("depth", depth_D)

        target = target.clip_to_image(remove_empty=True)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        if self.cfg["Pose"]:
            target.add_field("symmetry", self.symmetry)
            target.add_field("extents", self.extents)
            target.add_field("points", self.points)


        return img, target, idx
