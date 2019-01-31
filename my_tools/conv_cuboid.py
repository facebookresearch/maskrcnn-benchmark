"""
Get the cuboid vertices of an object
"""

import cv2
import numpy as np
import numpy.random as npr
import os
import os.path as osp
# import open3d
from transforms3d.quaternions import quat2mat, mat2quat

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from conv_depth import FATDataLoader, FT
from conv_test import get_cuboid_from_min_max, draw_cuboid_2d, get_random_color


class FATDataLoader4(FATDataLoader):
    def __init__(self, root_dir, ann_file, shuffle=True):
        super(FATDataLoader4, self).__init__(root_dir, ann_file, shuffle=shuffle)

        points_file = osp.join(self.root, "../points_all_orig.npy")
        self.points = np.load(points_file)
        assert (len(self.points) > 0)
        self.points_min = np.zeros((len(self.points), 3), dtype=np.float32)
        self.points_max = self.points_min.copy()
        for ix,pts in enumerate(self.points):
            if len(pts) > 0:
                self.points_min[ix] = np.min(pts, axis=0)
                self.points_max[ix] = np.max(pts, axis=0)

    def next_batch(self, batch_sz, max_pad=50, random_crop=False):
        perm = self.get_next_batch_perm(batch_sz)
        annots = [self.annots[idx] for idx in perm]

        image_list = []
        labels_list = []
        cuboids_list = []

        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion

        for ann in annots:
            img_id = ann['image_id']
            cls = ann['category_id']
            img_data = self.images[self.img_index[img_id]]

            # load img
            img_file = img_data["file_name"]
            img_file_path = osp.join(self.root, img_file)
            img = cv2.imread(img_file_path)
            if img is None:
                print("Could not read %s"%(img_file_path))

            # get pose and derive cuboid from pose 
            meta = ann['meta']
            pose = meta['pose']  # qw,qx,qy,qz,x,y,z
            intrinsic_matrix = np.array(meta['intrinsic_matrix']).reshape((3,3))
            # bounds = meta['bounds']
            bounds = [self.points_min[cls], self.points_max[cls]]
            cuboid = get_cuboid_from_min_max(bounds[0], bounds[1])
            centroid = np.array(pose[4:])
            quat = pose[:4]
            R = quat2mat(quat)

            # get 2d projected cuboid
            cuboid_2d, _ = cv2.projectPoints(cuboid, R, centroid, intrinsic_matrix, dist_coeffs)
            cuboid_2d = cuboid_2d.squeeze()

            # # perform random crop centered on the object 
            # bbox = np.array(ann['bbox'])
            # get min max of proj cuboid
            cuboid_2d = np.round(cuboid_2d).astype(np.int32)
            x1,y1 = np.min(cuboid_2d, axis=0)
            x2,y2 = np.max(cuboid_2d, axis=0)

            ih,iw = img.shape[:2]
            min_x = max(0, x1 - max_pad)
            min_y = max(0, y1 - max_pad)
            max_x = min(iw, x2 + max_pad)
            max_y = min(ih, y2 + max_pad)

            qx = (max_x - min_x) // 3
            qy = (max_y - min_y) // 3
            if random_crop:
                x1 = npr.randint(min_x, min_x + qx)
                y1 = npr.randint(min_y, min_y + qy)
                x2 = npr.randint(max_x - qx, max_x)
                y2 = npr.randint(max_y - qy, max_y)
            else:
                x1 = min_x
                x2 = max_x
                y1 = min_y
                y2 = max_y

            cropped_img = img[y1:y2, x1:x2]
            cuboid_2d -= np.array([x1, y1])

            image_list.append(cropped_img)
            labels_list.append(cls)
            cuboids_list.append(cuboid_2d)

        return image_list, labels_list, cuboids_list, annots

    def visualize(self, data):
        image_list, labels_list, cuboids_list, annots = data
        unique_labels = np.unique(labels_list)
        color_dict = {l: get_random_color() for l in unique_labels}
        for ix,ann in enumerate(annots):
            img = image_list[ix].copy()
            label = labels_list[ix]
            color = color_dict[label]

            cuboid = cuboids_list[ix]
            for pt in cuboid:
                cv2.circle(img, tuple(pt), 3, color, -1)
            cv2.imshow("img", img)
            cv2.waitKey(0)

if __name__ == '__main__':

    dataset_dir = "./datasets/FAT"
    root_dir = osp.join(dataset_dir, "data")
    ann_file = osp.join(dataset_dir, "coco_fat_debug.json")

    data_loader = FATDataLoader4(root_dir, ann_file)
    data = data_loader.next_batch(8)
    data_loader.visualize(data)
    # data_loader.convert_data_batch_to_tensor(data)

