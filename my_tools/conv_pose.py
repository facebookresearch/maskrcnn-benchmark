"""
tries to regress quaternion using Shape match loss (ave dist loss) - uses rgb OR scaled depth as input
"""

# import json
import cv2
import numpy as np
import numpy.random as npr
import os
import os.path as osp
import open3d
from transforms3d.quaternions import quat2mat#, mat2quat
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from maskrcnn_benchmark.layers.ave_dist_loss import AverageDistanceLoss
from conv_test import ConvNet
from conv_depth import FATDataLoader, FT, create_cloud, backproject_camera, get_random_color

POSE_CHANNELS = 4

PIXEL_MEAN = [102.9801, 115.9465, 122.7717]

PoseNet = ConvNet
# class PoseNet(nn.Module):
#     def __init__(self, in_channels=3, out_channels=1):
#         super(PoseNet, self).__init__()

#         conv1_filters = 64
#         conv2_filters = 128
#         conv3_filters = 256

#         self.conv1 = nn.Conv2d(in_channels, conv1_filters, kernel_size=3, stride=2, padding=1)  # total stride = 2
#         self.conv2 = nn.Conv2d(conv1_filters, conv2_filters, kernel_size=3, stride=2, padding=1)  # total stride = 4
#         self.conv3 = nn.Conv2d(conv2_filters, conv3_filters, kernel_size=3, stride=2, padding=1)  # total stride = 8
#         self.bn1 = nn.BatchNorm2d(conv1_filters)
#         self.bn2 = nn.BatchNorm2d(conv2_filters)
#         self.bn3 = nn.BatchNorm2d(conv3_filters)

#         fc_channels = 1024
#         self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
#         self.poses_fc1 = nn.Linear(conv3_filters, fc_channels)
#         self.poses_fc2 = nn.Linear(fc_channels, fc_channels)
#         self.poses_fc3 = nn.Linear(fc_channels, out_channels)

#         self._init_params()

#     def _init_params(self):
#         conv_modules = [self.conv1, self.conv2, self.conv3]
#         for m in conv_modules:
#             nn.init.constant_(m.bias, 0)
#             nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

#         fc_modules = [self.poses_fc1, self.poses_fc2, self.poses_fc3]
#         for m in fc_modules:
#             nn.init.constant_(m.bias, 0)
#             nn.init.normal_(m.weight, mean=0, std=0.001)

#     def forward(self, x):
#         if len(x.shape) == 3:  # (N,H,W) to (N,1,H,W)
#             x = x.unsqueeze(1)
#         # c1 = F.relu(self.bn1(self.conv1(x)))
#         # c2 = F.relu(self.bn2(self.conv2(c1)))
#         # c3 = F.relu(self.bn3(self.conv3(c2)))
#         c1 = F.relu(self.conv1(x))
#         c2 = F.relu(self.conv2(c1))
#         c3 = F.relu(self.conv3(c2))

#         x = self.avgpool(c3)
#         x = x.view(x.size(0), -1)
#         fc1 = self.poses_fc1(x)
#         # fc1 = F.normalize(fc1, p=2, dim=1)
#         fc2 = self.poses_fc2(F.dropout(F.relu(fc1, inplace=True), 0.5, training=self.training))
#         fc3 = self.poses_fc3(F.dropout(F.relu(fc2, inplace=True), 0.5, training=self.training))

#         return torch.tanh(fc3)

class FATDataLoader2(FATDataLoader):
    def __init__(self, root_dir, ann_file, use_scaled_depth=False):
        super(FATDataLoader2, self).__init__(root_dir, ann_file, use_scaled_depth=use_scaled_depth)

        # load models as points (Class,N,3)
        models_dir = os.path.join(root_dir, "../models")
        _, self.points = self._load_object_points(models_dir)

        self.extents = np.max(self.points,axis=1) - np.min(self.points,axis=1)

        # load symmetry file
        symmetry_file = os.path.join(root_dir, "../symmetry.txt")
        self.symmetry = self._load_object_symmetry(symmetry_file)

        is_symmetric = True
        point_blob = self.points#.copy()
        for i in range(1, self.num_classes):
            # compute the rescaling factor for the points
            weight = 2.0 / np.amax(self.extents[i, :])
            weight = max(weight, 10.0)
            if self.symmetry[i] > 0: #and is_symmetric:
                weight *= 4
            point_blob[i, :, :] *= weight

    def _load_object_points(self, models_dir):

        points = [[] for _ in range(self.num_classes)]
        num = np.inf

        for i in range(1, self.num_classes):
            point_file = os.path.join(models_dir, self._classes[i], 'points.xyz')
            # print point_file
            assert os.path.exists(point_file), 'Path does not exist: {}'.format(point_file)
            points[i] = np.loadtxt(point_file)
            print("Loaded %s"%(point_file))
            if points[i].shape[0] < num:
                num = points[i].shape[0]

        points_all = np.zeros((self.num_classes, num, 3), dtype=np.float32)
        for i in range(1, self.num_classes):
            points_all[i, :, :] = points[i][:num, :]

        return points, points_all

    def _load_object_symmetry(self, symmetry_file):

        assert os.path.exists(symmetry_file), \
                'Path does not exist: {}'.format(symmetry_file)

        symmetry = np.zeros((self.num_classes), dtype=np.float32)
        symmetry[1:] = np.loadtxt(symmetry_file)

        return symmetry

    def next_batch(self, batch_sz, add_noise=False):
        data = super(FATDataLoader2, self).next_batch(batch_sz)
        image_list, labels_list, mask_list, depth_list, annots = data
        pose_list = []
        for ix,ann in enumerate(annots):
            meta = ann['meta']
            pose = meta['pose']  # qw,qx,qy,qz,x,y,z

            quat = pose[:4]

            if add_noise:
                depth = depth_list[ix]
                sz = depth.shape
                mean = 0.0
                std = 0.1
                noise = np.random.normal(mean, std, size=sz)
                depth_list[ix] = depth + noise

            pose_list.append(quat)
        return [image_list, labels_list, mask_list, depth_list, pose_list, annots]

    def convert_data_batch_to_tensor(self, data, resize_shape=56, use_cuda=False):
        image_list, labels_list, mask_list, depth_list, pose_list, annots = data

        # normalize the pixels by pixel mean
        image_list2 = copy.copy(data[0])
        for ix,im in enumerate(image_list2):
            image_list2[ix] = im.astype(np.float32) - PIXEL_MEAN

        data2 = [image_list2, labels_list, mask_list, depth_list, annots]
        t_image_tensor, t_labels_tensor, t_mask_tensor, t_depth_tensor = \
                super(FATDataLoader2, self).convert_data_batch_to_tensor(data2, resize_shape, use_cuda)
        t_pose_tensor = FT(pose_list)
        if use_cuda:
            t_pose_tensor = t_pose_tensor.cuda()

        return t_image_tensor, t_labels_tensor, t_mask_tensor, t_depth_tensor, t_pose_tensor

    def get_object_points(self):
        return self.points.copy(), self.symmetry.copy()

def get_4x4_transform(pose):
    object_pose_matrix4f = np.identity(4)
    object_pose = np.array(pose)
    if object_pose.shape == (4,4):
        object_pose_matrix4f = object_pose
    elif object_pose.shape == (3,4):
        object_pose_matrix4f[:3,:] = object_pose
    elif len(object_pose) == 7:
        object_pose_matrix4f[:3,:3] = quat2mat(object_pose[:4])
        object_pose_matrix4f[:3,-1] = object_pose[4:]    
    else:
        print("[WARN]: Object pose is not of shape (4,4) or (3,4) or 1-d quat (7), skipping...")
    return object_pose_matrix4f

def render_object_pose(im, depth, labels, meta_data, pose_data, points):
    """
    im: rgb image of the scene
    depth: depth image of the scene
    meta_data: dict({'intrinsic_matrix': K, 'factor_depth': })
    pose_data: [{"name": "004_sugar_box", "pose": 3x4 or 4x4 matrix}, {...}, ]
    """
    if len(pose_data) == 0:
        return 

    rgb = im.copy()
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32)[:,:,::-1] / 255

    intrinsics = meta_data['intrinsic_matrix']
    factor_depth = meta_data['factor_depth']

    X = backproject_camera(depth, intrinsics, factor_depth)
    cloud_rgb = rgb # .astype(np.float32)[:,:,::-1] / 255
    cloud_rgb = cloud_rgb.reshape((cloud_rgb.shape[0]*cloud_rgb.shape[1],3))
    scene_cloud = create_cloud(X.T, colors=cloud_rgb)

    if len(pose_data) == 0:
        open3d.draw_geometries([scene_cloud])
        return 

    all_objects_cloud = open3d.PointCloud()
    for ix,pd in enumerate(pose_data):
        object_cls = labels[ix]
        object_pose = pd
        object_pose_matrix4f = get_4x4_transform(object_pose)

        object_pts3d = points[object_cls] # read_xyz_file(object_cloud_file)
        pt_colors = np.zeros(object_pts3d.shape, np.float32)
        pt_colors[:] = np.array(get_random_color()) / 255
        object_cloud = create_cloud(object_pts3d, colors=pt_colors, T=object_pose_matrix4f)
        # object_cloud.transform(object_pose_matrix4f)
        all_objects_cloud += object_cloud

        # print("Showing %s"%(object_name))
    open3d.draw_geometries([scene_cloud, all_objects_cloud])


def train(model, dg, use_rgb=False, n_iters=1000, lr=1e-3):

    from tensorboardX import SummaryWriter
    writer = SummaryWriter()

    # epochs = 10
    batch_size = 32

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))


    loss_type = "ave_dist"
    loss_fn = AverageDistanceLoss(margin=0.01)

    points, symmetry = dg.get_object_points()
    points_tensor = FT(points).cuda()
    symmetry_tensor = FT(symmetry).cuda()

    all_losses = []
    losses = []
    for iter in range(n_iters):
        data = dg.next_batch(batch_size, add_noise=True)

        image_tensor, labels_tensor, mask_tensor, depth_tensor, pose_tensor = \
                dg.convert_data_batch_to_tensor(data, use_cuda=True)
        optimizer.zero_grad()

        if not use_rgb:
            input_tensor = depth_tensor
            input_tensor[mask_tensor==0] = 0
        else:
            input_tensor = image_tensor
        pose_pred = model(input_tensor)

        if pose_tensor.numel() == 0 or pose_pred.numel() == 0:
            continue

        pose_pred = torch.tanh(pose_pred)
        # N = outputs.size(0)
        # channels = outputs.size(1) / POSE_CHANNELS

        pp_size = pose_pred.shape
        N,C = pp_size
        pp = pose_pred.view(N, -1, POSE_CHANNELS)  # N,classes,4
        pos_inds = torch.arange(N)
        pp = pp[pos_inds, labels_tensor]  # N,4

        # loss
        loss = loss_fn(pp, pose_tensor, labels_tensor, points_tensor, symmetry_tensor)
        loss.backward()
        optimizer.step()
        loss_value = loss.item()

        losses.append(loss_value)
        all_losses.append(loss_value)

        writer.add_scalar('data/pose_loss', loss_value, iter)

        if iter % 20 == 0 and iter > 0:
            print("iter %d of %d -> (%s) Total loss: %.4f, Avg loss: %.4f"%(iter, n_iters, loss_type, np.mean(losses), np.mean(all_losses)))
            losses = []

    print("iter %d of %d -> Total loss: %.4f, Avg loss: %.4f"%(iter, n_iters, np.mean(losses), np.mean(all_losses)))
    writer.close()

def test(model, dg, batch_sz = 8, use_rgb=False):
    model.eval()

    intrinsics = np.array([[768.1605834960938, 0.0, 480.0], [0.0, 768.1605834960938, 270.0], [0.0, 0.0, 1.0]])
    factor_depth = 10000
    data = dg.next_batch(batch_sz)
    image_list, labels_list, mask_list, depth_list, pose_list, annots = data

    # convert to tensors
    image_tensor, labels_tensor, mask_tensor, depth_tensor, pose_tensor = \
            dg.convert_data_batch_to_tensor(data, use_cuda=True)

    if not use_rgb:
        input_tensor = depth_tensor
        input_tensor[mask_tensor==0] = 0
    else:
        input_tensor = image_tensor

    preds = model(input_tensor)
    preds = torch.tanh(preds)
    preds = preds.detach().cpu().numpy()

    points_file = os.path.join(dg.root, "../points_all_orig.npy")
    points = np.load(points_file)

    meta_data = {'intrinsic_matrix': intrinsics.tolist(), 'factor_depth': factor_depth}

    for ix,pose_pred in enumerate(preds):
        ann = annots[ix]
        img_id = ann['image_id']
        cls = ann['category_id']

        pose_pred = pose_pred[cls*POSE_CHANNELS:(cls+1)*POSE_CHANNELS]

        img_data = dg.images[dg.img_index[img_id]]

        # load img
        img_file = img_data["file_name"]
        img_file_path = osp.join(dg.root, img_file)
        img = cv2.imread(img_file_path)
        if img is None:
            print("Could not read %s"%(img_file_path))
            continue

        # load depth
        depth_file = img_data['depth_file_name']
        depth_file_path = osp.join(dg.root, depth_file)
        depth = cv2.imread(depth_file_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            print("Could not read %s"%(depth_file_path))
            continue

        # add translation to quaternion
        meta = ann['meta']
        centroid = meta['centroid']
        final_pose = np.zeros(7, dtype=np.float32)
        final_pose[:4] = pose_pred
        final_pose[4:] = centroid

        label = labels_list[ix]
        render_object_pose(img, depth, [label], meta_data, [final_pose], points)


if __name__ == '__main__':

    CLASSES = [
        '__background__', '002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can',
         '006_mustard_bottle', \
         '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana',
         '019_pitcher_base', \
         '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors',
         '040_large_marker', \
         '051_large_clamp', '052_extra_large_clamp', '061_foam_brick'
    ]
    num_classes = len(CLASSES)

    USE_SCALED_DEPTH = True
    USE_RGB = True
    in_channels = 3 if USE_RGB else 1

    if USE_RGB:
        save_path = "./model_pose_rgb_0.pth"
    else:
        save_path = "./model_pose_0.pth"

    root_dir = "./datasets/FAT/data"
    ann_file = "./datasets/FAT/coco_fat_train_3500.json"

    print("USE_RGB: %s"%(USE_RGB))
    data_loader = FATDataLoader2(root_dir, ann_file, USE_SCALED_DEPTH)

    model = PoseNet(in_channels=in_channels, out_channels=num_classes * POSE_CHANNELS )
    model.cuda()
    model.load_state_dict(torch.load(save_path))
    print("Loaded %s"%(save_path))

    n_iters = 3000
    lr = 1e-3
    # train(model, data_loader, use_rgb=USE_RGB, n_iters=n_iters, lr=lr)
    # torch.save(model.state_dict(), save_path)

    test_ann_file = "./datasets/FAT/coco_fat_test_500.json"
    # test_ann_file = "./datasets/FAT/coco_fat_mixed_temple_1_n100.json"
    test_data_loader = FATDataLoader2(root_dir, test_ann_file, USE_SCALED_DEPTH)    
    test(model, test_data_loader, batch_sz=32, use_rgb=USE_RGB)
