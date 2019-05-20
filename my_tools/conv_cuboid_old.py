"""
Get the cuboid vertices of all objects in image
"""

import cv2
import numpy as np
# import numpy.random as npr
import os
import os.path as osp
# import open3d
from transforms3d.quaternions import quat2mat, mat2quat

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from conv_depth import FATDataLoader, FT, LT
from conv_test import get_cuboid_from_min_max, draw_cuboid_2d, get_random_color

PIXEL_MEAN = [102.9801, 115.9465, 122.7717]


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

        self.img_annot_map = defaultdict(list)
        for ix,ann in enumerate(self.annots):
            img_id = ann['image_id']
            self.img_annot_map[img_id].append(ix)

        self.total_cnt = self.total_images  # set total count to images
        self._reset_permutation()

    def next_batch(self, batch_sz):
        perm = self.get_next_batch_perm(batch_sz)
        images_data = [self.images[idx] for idx in perm]

        image_list = []
        labels_list = []
        cuboids_list = []

        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion

        for img_data in images_data:
            img_id = img_data['id']
            img_annots = self.img_annot_map[img_id]

            # load img
            img_file = img_data["file_name"]
            img_file_path = osp.join(self.root, img_file)
            img = cv2.imread(img_file_path)
            if img is None:
                print("Could not read %s"%(img_file_path))

            img_labels_list = []
            img_cuboids_list = []
            for ann_idx in img_annots:
                # get pose and derive cuboid from pose
                ann = self.annots[ann_idx]
                meta = ann['meta']
                pose = meta['pose']  # qw,qx,qy,qz,x,y,z
                intrinsic_matrix = np.array(meta['intrinsic_matrix']).reshape((3,3))
                cls = ann['category_id']
                # bounds = meta['bounds']
                bounds = [self.points_min[cls], self.points_max[cls]]
                cuboid = get_cuboid_from_min_max(bounds[0], bounds[1])
                centroid = np.array(pose[4:])
                quat = pose[:4]
                R = quat2mat(quat)

                # get 2d projected cuboid
                cuboid_2d, _ = cv2.projectPoints(cuboid, R, centroid, intrinsic_matrix, dist_coeffs)
                cuboid_2d = cuboid_2d.squeeze()

                img_labels_list.append(cls)
                img_cuboids_list.append(cuboid_2d)

            image_list.append(img)
            labels_list.append(img_labels_list)
            cuboids_list.append(img_cuboids_list)

        return image_list, labels_list, cuboids_list, images_data

    def visualize(self, data):
        image_list, labels_list, cuboids_list, _ = data
        # unique_labels = np.unique([l for ll in labels_list for l in ll])
        color_dict = {l: get_random_color() for l in range(self.num_classes)}
        for ix,img in enumerate(image_list):
            img = image_list[ix].copy()
            labels = labels_list[ix]
            cuboids = cuboids_list[ix]

            for jx,label in enumerate(labels):
                color = color_dict[label]
                cuboid = cuboids[jx]
                for pt in cuboid:
                    cv2.circle(img, tuple(pt), 3, color, -1)
            cv2.imshow("img", img)
            cv2.waitKey(0)

    def convert_data_batch_to_tensor(self, data, max_width=700, use_cuda=False):
        image_list, labels_list, cuboids_list, _ = data
        N = len(image_list)

        max_height = 0
        for im in image_list:
            h,w = im.shape[:2]
            assert w >= h
            ratio_w = float(max_width) / w
            h *= ratio_w
            h = int(h)
            if h > max_height:
                max_height = h 

        t_images = np.zeros((N, max_height, max_width, 3), dtype=np.float32)
        t_mask_cuboids = np.zeros((N, self.num_classes, max_height, max_width), dtype=np.float32)  # mask storing all the pixels
        # t_mask_weights = t_mask_cuboids.copy()

        for ix, im in enumerate(image_list):
            # compute resize ratio
            h,w = im.shape[:2]
            ratio = float(max_width) / w
            # ratio_h = int(h * ratio_w)
            resize_h = int(h * ratio)
            resize_w = max_width
            t_im = cv2.resize(im, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
            t_images[ix, :resize_h + 1, :resize_w + 1] = t_im
            t_m = t_mask_cuboids[ix]
            # t_mw = t_mask_weights[ix]
            labels = labels_list[ix]
            cuboids = cuboids_list[ix]

            for jx, cuboid in enumerate(cuboids):
                cuboid = cuboids[jx].copy()
                cuboid *= ratio
                cuboid = np.round(cuboid).astype(np.int32)
                cls = labels[jx]
                mm = t_m[cls]
                for pt in cuboid:
                    if pt[0] < resize_w and pt[1] < resize_h:
                        mm[pt[1], pt[0]] = 1
                # mw = t_mw[cls]
                # for pt in r_cuboids:
                #     cv2.circle

        t_images = np.transpose(t_images, [0,3,1,2])
        t_image_tensor = FT(t_images)
        t_mask_tensor = FT(t_mask_cuboids)
        # t_mask_w_tensor = FT(t_mask_weights)
        if use_cuda:
            t_image_tensor = t_image_tensor.cuda()
            t_mask_tensor = t_mask_tensor.cuda()
            # t_mask_w_tensor = t_mask_w_tensor.cuda()

        return t_image_tensor, t_mask_tensor#, t_mask_w_tensor

class ConvCuboid(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(ConvCuboid, self).__init__()

        conv1_filters = 64
        conv2_filters = 128
        conv3_filters = 256
        conv4_filters = 512
        conv5_filters = 1024

        self.conv1 = nn.Conv2d(in_channels, conv1_filters, kernel_size=5, stride=2, padding=5//2)
        self.conv2 = nn.Conv2d(conv1_filters, conv2_filters, kernel_size=5, stride=1, padding=5//2)
        self.conv3 = nn.Conv2d(conv2_filters, conv3_filters, kernel_size=5, stride=2, padding=5//2)
        self.conv4 = nn.Conv2d(conv3_filters, conv4_filters, kernel_size=3, stride=1, padding=3//2)
        self.conv5 = nn.Conv2d(conv4_filters, conv5_filters, kernel_size=3, stride=2, padding=3 // 2)
        self.conv6 = nn.Conv2d(conv5_filters, out_channels, kernel_size=3, stride=1, padding=3 // 2)
        self.bn1 = nn.BatchNorm2d(conv1_filters)
        self.bn2 = nn.BatchNorm2d(conv2_filters)
        self.bn3 = nn.BatchNorm2d(conv3_filters)
        self.bn4 = nn.BatchNorm2d(conv4_filters)
        self.bn5 = nn.BatchNorm2d(conv5_filters)
        self.bn6 = nn.BatchNorm2d(out_channels)

        self.max_pool = nn.MaxPool2d(2, stride=2)
        # self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)

        # self.reg = nn.Linear(conv5_filters, out_channels)
        # self.reg = nn.Conv2d(conv2_filters, out_channels, kernel_size=3, stride=1, padding=3 // 2)
        self._init_params()

    def _init_params(self):
        conv_modules = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6] #self.conv_t1, self.conv_t2, self.conv_t3]
        for m in conv_modules:
            nn.init.constant_(m.bias, 0)
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            # nn.init.normal_(m.weight, mean=0, std=0.001)

        # fc_modules = [self.reg]
        # for m in fc_modules:
        #     nn.init.normal_(m.weight, mean=0, std=0.01)
        #     nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # batch_sz = len(x)
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = self.max_pool(c1)
        c2 = F.relu(self.bn2(self.conv2(c1)))
        c3 = F.relu(self.bn3(self.conv3(c2)))
        c4 = F.relu(self.bn4(self.conv4(c3)))
        c5 = F.relu(self.bn5(self.conv5(c4)))
        c6 = self.bn6(self.conv6(c5))

        # x = self.avgpool(c5)
        # x = x.view(x.size(0), -1)
        # out = self.reg(x)
        return c6 # torch.tanh(out)

def resize_tensor(t, H, W, mode='bilinear'):
    return F.upsample(t, size=(H, W), mode=mode)

def train(model, dg, n_iters=100, lr=1e-3):

    # from tensorboardX import SummaryWriter
    # writer = SummaryWriter()

    batch_size = 2

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    loss_type = "dist"
    # loss_fn = euclidean_distance_loss

    all_losses = []
    losses = []
    for iter in range(n_iters):
        data = dg.next_batch(batch_size)

        # normalize the pixels by pixel mean
        image_list = data[0]
        for ix,im in enumerate(image_list):
            image_list[ix] = im.astype(np.float32) - PIXEL_MEAN

        image_tensor, mask_tensor = dg.convert_data_batch_to_tensor(data, use_cuda=True)
        optimizer.zero_grad()

        pred = model(image_tensor)
        pp_size = pred.shape
        N,C,H,W = pp_size  # N,classes,H,W
        pos_inds = torch.arange(N)

        gt = resize_tensor(mask_tensor, H, W, mode='nearest')  # N,classes,H,W

        # loss
        loss = loss_fn(pp, gt)
        loss.backward()
        optimizer.step()
        loss_value = loss.item()

        losses.append(loss_value)
        all_losses.append(loss_value)

        # writer.add_scalar('data/vf_loss', loss_value, iter)

        if iter % 20 == 0 and iter > 0:
            print("iter %d of %d -> (%s) Total loss: %.4f, Avg loss: %.4f"%(iter, n_iters, loss_type, np.mean(losses), np.mean(all_losses)))
            losses = []

    print("iter %d of %d -> Total loss: %.4f, Avg loss: %.4f"%(iter, n_iters, np.mean(losses), np.mean(all_losses)))
    # writer.close()


if __name__ == '__main__':

    dataset_dir = "./datasets/FAT"
    root_dir = osp.join(dataset_dir, "data")
    ann_file = osp.join(dataset_dir, "coco_fat_debug.json")

    data_loader = FATDataLoader4(root_dir, ann_file)
    data = data_loader.next_batch(2)
    # data_loader.visualize(data)
    t_image_tensor, t_mask_tensor = data_loader.convert_data_batch_to_tensor(data)
    num_classes = data_loader.num_classes

    model = ConvCuboid(3, out_channels=data_loader.num_classes)
    model.cuda()

    n_iters = 100
    lr = 1e-3
    train(model, data_loader, n_iters=n_iters, lr=lr)
