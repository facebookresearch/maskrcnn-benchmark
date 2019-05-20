"""
Get the cuboid vertices of an object as per pixel sigmoid
"""

import cv2
import numpy as np
import numpy.random as npr
import os
import os.path as osp
# import open3d
from transforms3d.quaternions import quat2mat, mat2quat
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from conv_depth import FATDataLoader, FT, LT
from conv_test import get_cuboid_from_min_max, draw_cuboid_2d, get_random_color, conv_transpose2d_by_factor, PIXEL_MEAN


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

    def next_batch(self, batch_sz, max_pad=50, random_crop=False, flip_prob=0.0):
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
            h, w = cropped_img.shape[:2]
            cuboid_2d -= np.array([x1, y1])

            rand = npr.rand(1)[0]
            if rand < flip_prob:
                if npr.rand(1)[0] < 0.5:
                    cropped_img = np.fliplr(cropped_img) # flip about vertical axis
                    cuboid_2d[:, 0] = w - 1 - cuboid_2d[:, 0]
                else:
                    cropped_img = np.flipud(cropped_img)  # flip about horizontal axis
                    cuboid_2d[:, 1] = h - 1 - cuboid_2d[:, 1]

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

    def convert_data_batch_to_tensor(self, data, resize_height=480, resize_width=800, radius=4, use_cuda=False):
        image_list, labels_list, cuboids_list, _ = data
        N = len(image_list)

        if N == 0:
            return []

        t_image_list = np.zeros((N, resize_height, resize_width, 3), dtype=np.float32)
        t_mask_list = np.zeros((N, resize_height, resize_width), dtype=np.float32)
        t_mask_weights_list = t_mask_list - 1  # set all to -1
        for ix, im in enumerate(image_list):
            
            h,w = im.shape[:2]
            t_im = cv2.resize(im, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
            t_im = t_im.astype(np.float32) / 255  # assumes 0-255!
            t_image_list[ix] = t_im

            factor_w = float(resize_width) / w
            factor_h = float(resize_height) / h
            t_mask = t_mask_list[ix]
            t_mask_weights = t_mask_weights_list[ix]

            cuboid = cuboids_list[ix].astype(np.float32)
            cuboid[:,0] *= factor_w
            cuboid[:,1] *= factor_h
            cuboid = np.round(cuboid).astype(np.int32)
            # set weights of pixels surrounding a keypoint to 0 (i.e. radius of 2 pix)
            # radius = 2
            for pt in cuboid:
                cv2.circle(t_mask_weights, tuple(pt), radius, 1, -1)
            for pt in cuboid:
                if pt[1] >= resize_height or pt[0] >= resize_width:
                    continue
                t_mask[pt[1],pt[0]] = 1
                t_mask_weights[pt[1],pt[0]] = 1

            total_neg = np.sum(t_mask_weights==-1)
            total_pos = np.sum(t_mask_weights == 1)
            t_mask_weights[t_mask_weights==-1] = 0.5 / total_neg
            t_mask_weights[t_mask_weights == 1] = 0.5 / total_pos

        t_image_list = np.transpose(t_image_list, [0,3,1,2])  # (N,H,W,3) to (N,3,H,W)
        t_image_tensor = FT(t_image_list)
        t_mask_tensor = FT(t_mask_list)
        t_mask_weights_tensor = FT(t_mask_weights_list)
        t_labels_tensor = LT(labels_list)
        if use_cuda:
            t_image_tensor = t_image_tensor.cuda()
            t_mask_tensor = t_mask_tensor.cuda()
            t_mask_weights_tensor = t_mask_weights_tensor.cuda()
            t_labels_tensor = t_labels_tensor.cuda()

        return t_image_tensor, t_labels_tensor, t_mask_tensor, t_mask_weights_tensor

class ConvNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(ConvNet, self).__init__()

        conv1_filters = 64
        conv2_filters = 128
        conv3_filters = 256

        self.conv1 = nn.Conv2d(in_channels, conv1_filters, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(conv1_filters, conv2_filters, kernel_size=5, stride=2, padding=1)
        self.conv3 = nn.Conv2d(conv2_filters, conv3_filters, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(conv1_filters)
        self.bn2 = nn.BatchNorm2d(conv2_filters)
        self.bn3 = nn.BatchNorm2d(conv3_filters)

        conv_t_filters = 512
        self.conv_t1 = conv_transpose2d_by_factor(conv3_filters, conv_t_filters, factor=2)
        self.conv_t2 = conv_transpose2d_by_factor(conv_t_filters, conv_t_filters, factor=2)
        self.conv_t3 = conv_transpose2d_by_factor(conv_t_filters, conv_t_filters, factor=2)
        self.reg = nn.Conv2d(conv_t_filters, 64, kernel_size=5, stride=1, padding=5 // 2)
        self.reg2 = nn.Conv2d(64, out_channels, kernel_size=5, stride=1, padding=5 // 2)

    def forward(self, x):
        # batch_sz = len(x)
        c1 = F.relu(self.bn1(self.conv1(x)))
        c2 = F.relu(self.bn2(self.conv2(c1)))
        c3 = F.relu(self.bn3(self.conv3(c2)))
        # c1 = F.relu(self.conv1(x))
        # c2 = F.relu(self.conv2(c1))
        # c3 = F.relu(self.conv3(c2))
        ct1 = F.relu(self.conv_t1(c3))
        ct2 = F.relu(self.conv_t2(ct1))
        ct3 = F.relu(self.conv_t3(ct2))
        out = F.relu(self.reg(ct3))
        out = self.reg2(out)
        return out


def train(model, dg, n_iters=1000, batch_sz = 4, lr=1e-3):
    # from tensorboardX import SummaryWriter
    # writer = SummaryWriter()

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    criterion = nn.BCELoss(reduce=False)

    all_losses = []
    losses = []
    flip_prob = 0.5
    for iter in range(n_iters):
        data = dg.next_batch(batch_sz, flip_prob=flip_prob)

        # normalize the pixels by pixel mean
        image_list = data[0]
        for ix,im in enumerate(image_list):
            image_list[ix] = im.astype(np.float32) - PIXEL_MEAN

        image_tensor, labels_tensor, mask_tensor, mask_weights_tensor = dg.convert_data_batch_to_tensor(data, 
                    resize_height=56, resize_width=56, radius=2, use_cuda=True)
        optimizer.zero_grad()

        pred = model(image_tensor)
        N = pred.size(0)

        if N == 0:
            continue

        pos_inds = torch.arange(N)
        pred = pred[pos_inds, labels_tensor]  # (N,H,W)
        pred = torch.sigmoid(pred) 

        # loss
        entropy_loss = criterion(pred, mask_tensor)
        weighted_loss = torch.mul(entropy_loss, mask_weights_tensor)
        loss = weighted_loss.sum(dim=(1,2)).mean()
        loss.backward()
        optimizer.step()
        loss_value = loss.item()

        losses.append(loss_value)
        all_losses.append(loss_value)

        # writer.add_scalar('data/cuboid_binary_entropy_loss', loss_value, iter)

        if iter % 20 == 0 and iter > 0:
            print("iter %d of %d -> Total loss: %.4f, Avg loss: %.4f"%(iter, n_iters, np.mean(losses), np.mean(all_losses)))
            losses = []

    print("iter %d of %d -> Total loss: %.4f, Avg loss: %.4f"%(iter, n_iters, np.mean(losses), np.mean(all_losses)))
    # writer.close()

def test(model, dg, batch_sz = 8):
    model.eval()

    data = dg.next_batch(batch_sz, flip_prob=0)
    image_list, labels_list, cuboids_list, annots = data

    # normalize the pixels by pixel mean
    image_list = data[0]
    ori_image_list = copy.copy(image_list)
    for ix,im in enumerate(image_list):
        image_list[ix] = im.astype(np.float32) - PIXEL_MEAN

    image_tensor, labels_tensor, mask_tensor, mask_weights_tensor = dg.convert_data_batch_to_tensor(data, resize_height=56, resize_width=56, radius=2, use_cuda=True)

    preds = model(image_tensor)
    preds = torch.sigmoid(preds).detach().cpu().numpy()#.squeeze()

    for ix,pred in enumerate(preds):
        img = ori_image_list[ix]
        cls = labels_list[ix]

        # pred
        img2 = img.copy()
        m = pred[cls]
        h,w = img.shape[:2]

        m = cv2.resize(m, (w,h), interpolation=cv2.INTER_LINEAR)
        m[m >= 0.5] = 1
        m[m < 0.5] = 0
        img2[m==1] = [0,255,0]
        cv2.imshow("pred", img2)

        # gt
        cuboid = cuboids_list[ix]
        for pt in cuboid:
            cv2.circle(img, tuple(pt), 3, (0,0,255), -1)
        cv2.imshow("gt", img)
        
        cv2.waitKey(0)


if __name__ == '__main__':

    dataset_dir = "./datasets/FAT"
    root_dir = osp.join(dataset_dir, "data")
    ann_file = osp.join(dataset_dir, "coco_fat_debug.json")
    # ann_file = osp.join(dataset_dir, "coco_fat_train_3500.json")

    data_loader = FATDataLoader4(root_dir, ann_file)
    data = data_loader.next_batch(8, flip_prob=0.5)
    data_loader.visualize(data)
    # data_loader.convert_data_batch_to_tensor(data, resize_height=56, resize_width=56)

    out_channels = data_loader.num_classes
    model = ConvNet(in_channels=3, out_channels=out_channels)
    model.cuda()

    save_path = "model_cuboids_0.pth"
    # model.load_state_dict(torch.load(save_path))
    # print("Loaded %s"%(save_path))

    n_iters = 100
    lr = 1e-3
    batch_sz = 32
    train(model, data_loader, n_iters=n_iters, batch_sz = batch_sz, lr=lr)
    # torch.save(model.state_dict(), save_path)

    # test_ann_file = osp.join(dataset_dir, "coco_fat_test_500.json")
    # test_data_loader = FATDataLoader4(root_dir, test_ann_file, shuffle=False)
    test_data_loader = data_loader
    test(model, test_data_loader, batch_sz=8)
