import torch

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
    vertex_centers = np.zeros((h,w,3),dtype=np.float32)
    # z = pose[2, 3]
    y, x = np.where(label_mask == 1)

    R = c - np.vstack((x, y))
    # compute the norm
    N = np.linalg.norm(R, axis=0) + 1e-10
    # normalization
    R = R / N # np.divide(R, np.tile(N, (2,1)))
    # assignment
    vertex_centers[y, x, 1] = R[1,:]
    vertex_centers[y, x, 0] = R[0,:]
    vertex_centers[y, x, 2] = z
    return vertex_centers

def _get_mask_from_polygon(polygons, im_size):
    width, height = im_size
    rles = mask_utils.frPyObjects(
        [p.numpy() for p in polygons], height, width
    )
    rle = mask_utils.merge(rles)
    mask = mask_utils.decode(rle)
    return mask

class COCOPoseDataset(COCODataset):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(COCOPoseDataset, self).__init__(ann_file, root, remove_images_without_annotations, transforms)

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        seg_mask_instance = [obj["segmentation"] for obj in anno]
        seg_mask_instance = SegmentationMask(seg_mask_instance, img.size)
        target.add_field("masks", seg_mask_instance)

        meta = [obj["meta"] for obj in anno]
        assert len(meta) == len(seg_mask_instance.polygons)
        # masks = [_get_mask_from_polygon(polygon, img.size) for polygon in seg_mask_instance.polygons]
        vertex_centers = []
        for ix, polygon_instance in enumerate(seg_mask_instance.polygons):
            center = meta[ix]['center']
            pose = meta[ix]['pose']
            z = np.log(pose[-1]) # z distance is the last value in the 3x4 transform matrix (index is -1 if matrix is a list)
            m = _get_mask_from_polygon(polygon_instance.polygons, img.size)
            vertex_centers.append(_generate_vertex_center_mask(m, center, z))
        vertex_centers = torch.Tensor(vertex_centers)
        vertexes = VertexMask(vertex_centers, img.size)
        target.add_field("vertex", vertexes)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx        
