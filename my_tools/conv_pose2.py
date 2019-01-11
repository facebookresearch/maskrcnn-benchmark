

"""
SAME AS conv_pose.py: tries to regress quaternion using Shape match loss (ave dist loss), but uses BOTH rgb and scaled depth as input
"""

# import json
import cv2
import numpy as np
import numpy.random as npr
import os
import os.path as osp
import open3d
from transforms3d.quaternions import quat2mat#, mat2quat

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from maskrcnn_benchmark.layers.ave_dist_loss import AverageDistanceLoss
from conv_pose import FATDataLoader2, FT, create_cloud, backproject_camera, get_random_color, render_object_pose

POSE_CHANNELS = 4

PIXEL_MEAN = [102.9801, 115.9465, 122.7717]

class PoseNet2(nn.Module):
    def __init__(self, num_classes=1):
        super(PoseNet2, self).__init__()

        conv1_filters = 64
        conv2_filters = 128
        conv3_filters = 256

        # for rgb
        self.conv1 = nn.Conv2d(3, conv1_filters, kernel_size=5, stride=2, padding=5//2)  # total stride = 2
        self.conv2 = nn.Conv2d(conv1_filters, conv2_filters, kernel_size=5, stride=2, padding=5//2)  # total stride = 4
        self.conv3 = nn.Conv2d(conv2_filters, conv3_filters, kernel_size=3, stride=2, padding=3//2)  # total stride = 8
        self.bn1 = nn.BatchNorm2d(conv1_filters)
        self.bn2 = nn.BatchNorm2d(conv2_filters)
        self.bn3 = nn.BatchNorm2d(conv3_filters)

        self.max_pool = nn.MaxPool2d(3, stride=2)

        # for depth
        self.dconv1 = nn.Conv2d(1, conv1_filters, kernel_size=5, stride=2, padding=5//2)  # total stride = 2
        self.dconv2 = nn.Conv2d(conv1_filters, conv2_filters, kernel_size=5, stride=2, padding=5//2)  # total stride = 4
        self.dconv3 = nn.Conv2d(conv2_filters, conv3_filters, kernel_size=3, stride=2, padding=3//2)  # total stride = 8

        fc_channels = 1024
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
        self.poses_fc1 = nn.Linear(conv3_filters, fc_channels)
        self.poses_fc2 = nn.Linear(fc_channels, fc_channels)
        self.poses_fc3 = nn.Linear(fc_channels, num_classes * POSE_CHANNELS)

        self._init_params()

    def _init_params(self):
        conv_modules = [self.conv1, self.conv2, self.conv3, self.dconv1, self.dconv2, self.dconv3]
        for m in conv_modules:
            nn.init.constant_(m.bias, 0)
            nn.init.kaiming_normal_(m.weight, mode="fan_out")#, nonlinearity="relu")
            # nn.init.normal_(m.weight, mean=0, std=0.001)

        fc_modules = [self.poses_fc1, self.poses_fc2, self.poses_fc3]
        for m in fc_modules:
            nn.init.constant_(m.bias, 0)
            nn.init.normal_(m.weight, mean=0, std=0.001)

    def forward(self, rgb, depth):
        if len(depth.shape) == 3:  # (N,H,W) to (N,1,H,W)
            depth = depth.unsqueeze(1)

        c1 = F.relu(self.bn1(self.conv1(rgb)))
        c2 = F.relu(self.bn2(self.conv2(c1)))
        c3 = F.relu(self.bn3(self.conv3(c2)))
        # c1 = F.relu(self.conv1(rgb))
        # c2 = F.relu(self.conv2(c1))
        # c3 = F.relu(self.conv3(c2))

        # d1 = F.relu(self.bn1(self.dconv1(depth)))
        # d2 = F.relu(self.bn2(self.dconv2(d1)))
        # d3 = F.relu(self.bn3(self.dconv3(d2)))
        d1 = F.relu(self.dconv1(depth))
        d2 = F.relu(self.dconv2(d1))
        d3 = F.relu(self.dconv3(d2))

        # # pooling
        # c3 = self.max_pool(c3)
        # d3 = self.max_pool(d3)

        x = torch.add(c3, d3)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        fc1 = self.poses_fc1(x)
        fc2 = self.poses_fc2(F.dropout(F.relu(fc1, inplace=True), 0.5, training=self.training))
        fc3 = self.poses_fc3(F.dropout(F.relu(fc2, inplace=True), 0.5, training=self.training))
        # fc3 = self.poses_fc3(F.relu(fc2, inplace=True))

        return torch.tanh(fc3)

def train(model, dg, n_iters=1000, lr=1e-3):

    from tensorboardX import SummaryWriter
    writer = SummaryWriter()

    # epochs = 10
    batch_size = 32

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)#, betas=(0.9, 0.999))
    # optimizer = optim.SGD(model.parameters(), lr=lr*10, momentum=0.9)#, betas=(0.9, 0.999))

    loss_type = "ave_dist"
    loss_fn = AverageDistanceLoss(margin=0.01)

    points, symmetry = dg.get_object_points()
    points_tensor = FT(points).cuda()
    symmetry_tensor = FT(symmetry).cuda()

    all_losses = []
    losses = []
    for iter in range(n_iters):
        data = dg.next_batch(batch_size, add_noise=True)

        # normalize the pixels by pixel mean
        image_list = data[0]
        for ix,im in enumerate(image_list):
            image_list[ix] = im.astype(np.float32) - PIXEL_MEAN
        # data[0] = image_list

        image_tensor, labels_tensor, mask_tensor, depth_tensor, pose_tensor = \
                dg.convert_data_batch_to_tensor(data, use_cuda=True)
        optimizer.zero_grad()

        depth_tensor[mask_tensor==0] = 0

        pose_pred = model(image_tensor, depth_tensor)

        if pose_tensor.numel() == 0 or pose_pred.numel() == 0:
            continue

        # N = outputs.size(0)
        # channels = outputs.size(1) / POSE_CHANNELS

        pp_size = pose_pred.shape
        N,C = pp_size
        pp = pose_pred.view(N, -1, POSE_CHANNELS)  # N,classes,4
        pos_inds = torch.arange(N)
        pp = pp[pos_inds, labels_tensor]  # N,4

        # loss
        loss = loss_fn(pp, pose_tensor, labels_tensor, points_tensor, symmetry_tensor)
        loss_value = loss.item()

        loss *= 2
        loss.backward()
        optimizer.step()

        losses.append(loss_value)
        all_losses.append(loss_value)

        writer.add_scalar('data/pose_loss', loss_value, iter)

        if iter % 20 == 0 and iter > 0:
            print("iter %d of %d -> (%s) Total loss: %.4f, Avg loss: %.4f"%(iter, n_iters, loss_type, np.mean(losses), np.mean(all_losses)))
            losses = []

        if iter % 5000 == 0 and iter > 0:
            torch.save(model.state_dict(), "./tmp_model_pose2_%d.pth"%(iter))

    print("iter %d of %d -> Total loss: %.4f, Avg loss: %.4f"%(iter, n_iters, np.mean(losses), np.mean(all_losses)))
    writer.close()

def test(model, dg, batch_sz = 8):
    model.eval()

    intrinsics = np.array([[768.1605834960938, 0.0, 480.0], [0.0, 768.1605834960938, 270.0], [0.0, 0.0, 1.0]])
    factor_depth = 10000
    data = dg.next_batch(batch_sz)
    image_list, labels_list, mask_list, depth_list, pose_list, annots = data

    image_list = data[0]
    for ix, im in enumerate(image_list):
        image_list[ix] = im.astype(np.float32) - PIXEL_MEAN

    # convert to tensors
    image_tensor, labels_tensor, mask_tensor, depth_tensor, pose_tensor = \
            dg.convert_data_batch_to_tensor(data, use_cuda=True)

    depth_tensor[mask_tensor == 0] = 0

    preds = model(image_tensor, depth_tensor)
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

    save_path = "./model_pose2.pth"

    root_dir = "./datasets/FAT/data"
    ann_file = "./datasets/FAT/coco_fat_mixed_temple_0.json"

    data_loader = FATDataLoader2(root_dir, ann_file, USE_SCALED_DEPTH)

    model = PoseNet2(num_classes=num_classes)
    model.cuda()
    model.load_state_dict(torch.load(save_path))
    print("Loaded %s"%(save_path))

    # n_iters = 20000
    # lr = 1e-3
    # train(model, data_loader, n_iters=n_iters, lr=lr)
    # torch.save(model.state_dict(), save_path)

    # test_ann_file = "./datasets/FAT/coco_fat_debug_200.json"
    test_ann_file = "./datasets/FAT/coco_fat_mixed_temple_1_n100.json"
    test_data_loader = FATDataLoader2(root_dir, test_ann_file, USE_SCALED_DEPTH)    
    # test_data_loader = data_loader
    test(model, test_data_loader, batch_sz=16)
