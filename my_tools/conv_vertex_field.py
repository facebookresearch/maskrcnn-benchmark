"""
Fully conv regression on 3D angular vertex fields
"""

import cv2
import numpy as np
# import numpy.random as npr
import os
import os.path as osp
# import open3d
from transforms3d.quaternions import quat2mat, mat2quat

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from conv_depth import FATDataLoader, FT, conv_transpose2d_by_factor, get_random_color, create_cloud#, backproject_camera,
from conv_test import ConvNet, get_4x4_transform, average_point_distance_metric
from conv_pose import render_object_pose

CHANNELS = 9
STRIDE = 8

PIXEL_MEAN = [102.9801, 115.9465, 122.7717]

BLUE = (255,0,0)
GREEN = (0,255,0)
RED = (0,0,255)

def smooth_l1(n, beta=1. / 9, size_average=True):
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()

def euclidean_distance_loss(input, target):
    N = input.size(0)
    assert target.shape == (N,3,3)
    dist_loss = torch.norm(input - target, dim=1)
    loss = smooth_l1(dist_loss, size_average=True)
    # loss = 0
    # for i in range(N):
    #     gt = target[i]  # 3,3
    #     pred = input[i] # 3,3
    #     dist_loss = torch.norm(pred - gt, dim=1)
    #     loss += smooth_l1(dist_loss, size_average=True)
    # loss = loss / N
    return loss

def resize_tensor(t, H, W, mode='bilinear'):
    return F.upsample(t, size=(H, W), mode=mode)

def get_cuboid_from_min_max(min_pt, max_pt):
    assert len(min_pt) == 3 and len(max_pt) == 3
    return np.array([
        [min_pt[0], min_pt[1], min_pt[2]],
        [min_pt[0], max_pt[1], min_pt[2]],
        [min_pt[0], min_pt[1], max_pt[2]],
        [min_pt[0], max_pt[1], max_pt[2]],
        [max_pt[0], min_pt[1], min_pt[2]],
        [max_pt[0], max_pt[1], min_pt[2]],
        [max_pt[0], min_pt[1], max_pt[2]],
        [max_pt[0], max_pt[1], max_pt[2]]
    ])

    # lines = [[0,1],[0,2],[1,3],[2,3],
    #      [4,5],[4,6],[5,7],[6,7],
    #      [0,4],[1,5],[2,6],[3,7]]


def draw_cuboid_2d(img2, cuboid, color):
    assert len(cuboid) == 8
    points = [tuple(pt) for pt in cuboid]
    for ix in range(len(points)):
        pt = points[ix]
        cv2.putText(img2, "%d"%(ix), pt, cv2.FONT_HERSHEY_COMPLEX, 0.4, color)
        cv2.circle(img2, pt, 2, (0,255,0), -1)

    lines = [[0,1],[0,2],[1,3],[2,3],
         [4,5],[4,6],[5,7],[6,7],
         [0,4],[1,5],[2,6],[3,7]]

    for line in lines:
        pt1 = points[line[0]]
        pt2 = points[line[1]]
        cv2.line(img2, pt1, pt2, color)

def draw_axis_pose(img, rvec, centroid, intrinsic_matrix, dist_coeffs=np.zeros((4,1))):
    """
    rvec: (3,3) rotation matrix
    centroid: (3) vector
    intrinsic_matrix: (3,3) matrix
    """
    # draw 3 axis pose
    # from https://github.com/jerryhouuu/Face-Yaw-Roll-Pitch-from-Pose-Estimation-using-OpenCV/blob/master/pose_estimation.py

    axis_vectors = rvec.T
    avec = axis_vectors * 0.1  # convert from metres (100cm) to 10 cm
    points = np.vstack((avec, centroid))
    imgpts, _ = cv2.projectPoints(points, np.identity(3), centroid, intrinsic_matrix, dist_coeffs)
    imgpts = np.round(imgpts).astype(np.int32)

    im_copy2 = img.copy()
    center_pt = tuple(imgpts[-1].ravel())
    cv2.line(im_copy2, center_pt, tuple(imgpts[0].ravel()), (255, 0, 0), 3)  # BLUE
    cv2.line(im_copy2, center_pt, tuple(imgpts[1].ravel()), (0, 255, 0), 3)  # GREEN
    cv2.line(im_copy2, center_pt, tuple(imgpts[2].ravel()), (0, 0, 255), 3)  # RED
    return im_copy2

class FATDataLoader3(FATDataLoader):
    def __init__(self, root_dir, ann_file, use_scaled_depth=False, shuffle=True):
        super(FATDataLoader3, self).__init__(root_dir, ann_file, use_scaled_depth=use_scaled_depth, shuffle=shuffle)

    def next_batch(self, batch_sz):
        data = super(FATDataLoader3, self).next_batch(batch_sz)
        image_list, labels_list, mask_list, depth_list, annots = data
        pose_list = []
        vertex_field_list = []

        # axis_vectors = np.identity(3) 

        for ix,ann in enumerate(annots):
            meta = ann['meta']
            pose = meta['pose']  # qw,qx,qy,qz,x,y,z

            quat = pose[:4]
            R = quat2mat(quat)

            t_vectors = R.T # np.dot(R, axis_vectors).T
            # x_axis = t_vectors[0]
            # y_axis = t_vectors[1]
            # z_axis = t_vectors[2]

            vertex_field_list.append(t_vectors)
        return [image_list, labels_list, mask_list, depth_list, vertex_field_list, annots]

    def convert_data_batch_to_tensor(self, data, resize_shape=56, use_cuda=False):
        image_list, labels_list, mask_list, depth_list, vertex_field_list, annots = data
        N = len(image_list)

        if N == 0:
            return []

        data2 = [image_list, labels_list, mask_list, depth_list, annots]
        t_image_tensor, t_labels_tensor, t_mask_tensor, t_depth_tensor = \
                super(FATDataLoader3, self).convert_data_batch_to_tensor(data2, resize_shape, use_cuda)
        
        # sz = resize_shape
        # C = vertex_field_list[0].size  # 3x3
        # C = CHANNELS
        # t_verts = np.zeros((N, sz, sz, C), dtype=np.float32) 
        # for ix, im in enumerate(image_list):
        #     vf = vertex_field_list[ix].flatten()  # 3x3
        #     t_verts[ix, :, :] = vf

        # # t_verts = t_verts.reshape((N,sz,sz,3,3))
        # t_verts = np.transpose(t_verts, [0,3,1,2])  # to pytorch format  (N,C,H,W)
        # t_vert_tensor = FT(t_verts)
        t_vert_tensor = FT(vertex_field_list)
        if use_cuda:
            t_vert_tensor = t_vert_tensor.cuda()

        return t_image_tensor, t_labels_tensor, t_mask_tensor, t_depth_tensor, t_vert_tensor

    def visualize(self, data):
        image_list, labels_list, mask_list, depth_list, vertex_field_list, annots = data

        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion

        for ix,ann in enumerate(annots):
            # img = image_list[ix]
            img_id = ann['image_id']
            img_data = self.images[self.img_index[img_id]]
            # load img
            img_file = img_data["file_name"]
            img_file_path = osp.join(self.root, img_file)
            img = cv2.imread(img_file_path)
            if img is None:
                print("Could not read %s"%(img_file_path))

            meta = ann['meta']
            pose = meta['pose']  # qw,qx,qy,qz,x,y,z
            intrinsic_matrix = np.array(meta['intrinsic_matrix']).reshape((3,3))

            centroid = np.array(pose[4:])

            im_copy2 = draw_axis_pose(img, vertex_field_list[ix].T, centroid, intrinsic_matrix)

            cv2.imshow("im", im_copy2)
            cv2.waitKey(0)

def train(model, dg, n_iters=1000, lr=1e-3):

    from tensorboardX import SummaryWriter
    writer = SummaryWriter()

    batch_size = 32

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    loss_type = "dist"
    loss_fn = euclidean_distance_loss

    all_losses = []
    losses = []
    for iter in range(n_iters):
        data = dg.next_batch(batch_size)

        # normalize the pixels by pixel mean
        image_list = data[0]
        for ix,im in enumerate(image_list):
            image_list[ix] = im.astype(np.float32) - PIXEL_MEAN

        image_tensor, labels_tensor, mask_tensor, depth_tensor, vert_field_tensor = \
                dg.convert_data_batch_to_tensor(data, use_cuda=True)
        optimizer.zero_grad()

        vf_pred = model(image_tensor)

        if vert_field_tensor.numel() == 0 or vf_pred.numel() == 0:
            continue

        vf_pred = torch.tanh(vf_pred)

        pp_size = vf_pred.shape
        N,C = pp_size  # N,classes*9
        pp = vf_pred.view(N, -1, 3, 3)  # N,classes,3,3
        pos_inds = torch.arange(N)
        pp = pp[pos_inds, labels_tensor]  # N,3,3

        # # mask = F.interpolate(mask_tensor.unsqueeze(1), size=(H,W), mode='bilinear')
        # # pp = vf_pred.view(N, classes, CHANNELS, H*W)  # N,cls,9,H*W
        # pos_inds = torch.arange(N)
        # pp = pp[pos_inds, labels_tensor]  # N,9,H*W
        # # pp1 = pp.permute(0,2,1).clone()  # N,H*W,9
        # # pp1 = pp1.view(N, -1, 3, 3) # N,H*W,3,3
        # # mask = mask.view(N, -1) # N,H*W
        # # pp1[mask_tensor==0]
        target = vert_field_tensor  # N,3,3

        # loss
        loss = loss_fn(pp, target)
        loss.backward()
        optimizer.step()
        loss_value = loss.item()

        losses.append(loss_value)
        all_losses.append(loss_value)

        writer.add_scalar('data/vf_loss', loss_value, iter)

        if iter % 20 == 0 and iter > 0:
            print("iter %d of %d -> (%s) Total loss: %.4f, Avg loss: %.4f"%(iter, n_iters, loss_type, np.mean(losses), np.mean(all_losses)))
            losses = []

    print("iter %d of %d -> Total loss: %.4f, Avg loss: %.4f"%(iter, n_iters, np.mean(losses), np.mean(all_losses)))
    writer.close()

def test(model, dg, batch_sz = 8, visualize=True):
    model.eval()

    points_file = osp.join(dg.root, "../points_all_orig.npy")
    points = np.load(points_file)
    points_min = np.zeros((len(points), 3), dtype=np.float32)
    points_max = points_min.copy()
    for ix,pts in enumerate(points):
        if len(pts) > 0:
            points_min[ix] = np.min(pts, axis=0)
            points_max[ix] = np.max(pts, axis=0)

    data = dg.next_batch(batch_sz)
    image_list, labels_list, mask_list, depth_list, vert_field_list, annots = data

    # normalize by pixel mean
    image_list = data[0]
    for ix, im in enumerate(image_list):
        image_list[ix] = im.astype(np.float32) - PIXEL_MEAN

    # convert to tensors
    image_tensor, labels_tensor, mask_tensor, depth_tensor, vert_field_tensor = \
            dg.convert_data_batch_to_tensor(data, use_cuda=True)

    preds = model(image_tensor)
    preds = torch.tanh(preds)
    preds = preds.detach().cpu().numpy()

    dist_coeffs = np.zeros((4,1))

    metric_list = dict((ix, []) for ix,k in enumerate(dg._classes))
    for ix,pred in enumerate(preds):
        ann = annots[ix]
        cls = ann['category_id']
        assert cls == labels_list[ix]

        if pred.size == 0:
            continue
        C = len(pred)
        pred = pred.reshape((C//CHANNELS,CHANNELS))  # classes, 9
        pred = pred[cls]  #9
        pred_R = pred.reshape((3,3)).T

        gt_R = vert_field_list[ix].T
        axis_symmetry = dg.SYMMETRIES[cls]
        is_symmetric = np.sum(axis_symmetry) >= 2
        ave_dd = average_point_distance_metric(points[cls], pred_R, gt_R, closest_point=is_symmetric)
        ave_dd2 = average_point_distance_metric(points[cls], pred_R, gt_R, closest_point=True)
        print("%d) [cls %d (%s)] ADD: %.3f, ADD 2: %.3f"%(ix + 1, cls, dg._classes[cls], ave_dd, ave_dd2))
        metric_list[cls].append([ave_dd, ave_dd2])
        if not visualize:
            continue

        # print(axis_symmetry)
        # if np.sum(axis_symmetry) == 0:
        #     continue

        # load img
        img_id = ann['image_id']
        img_data = dg.images[dg.img_index[img_id]]
        img_file = img_data["file_name"]
        img_file_path = osp.join(dg.root, img_file)
        print(img_file_path)
        img = cv2.imread(img_file_path)
        if img is None:
            print("Could not read %s"%(img_file_path))
            continue

        meta = ann['meta']
        pose = meta['pose']  # qw,qx,qy,qz,x,y,z
        intrinsic_matrix = np.array(meta['intrinsic_matrix']).reshape((3,3))
        # bounds = meta['bounds']
        bounds = [points_min[cls], points_max[cls]]
        cuboid = get_cuboid_from_min_max(bounds[0], bounds[1])
        centroid = np.array(pose[4:])

        cuboid_2d, _ = cv2.projectPoints(cuboid, gt_R, centroid, intrinsic_matrix, dist_coeffs)
        cuboid_2d = np.round(cuboid_2d).squeeze().astype(np.int32)
        img_copy = img.copy()
        draw_cuboid_2d(img_copy, cuboid_2d, (255,0,0))

        im_copy2 = draw_axis_pose(img_copy, pred_R, centroid, intrinsic_matrix)
        im_copy3 = draw_axis_pose(img_copy, gt_R, centroid, intrinsic_matrix)

        cv2.imshow("pred_poses", im_copy2)
        cv2.imshow("gt_poses", im_copy3)
        cv2.waitKey(0)

        # load depth
        depth_file = img_data['depth_file_name']
        depth_file_path = osp.join(dg.root, depth_file)
        depth = cv2.imread(depth_file_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            print("Could not read %s"%(depth_file_path))
            continue

        # add translation to quaternion
        centroid = meta['centroid']
        final_pose = np.zeros(7, dtype=np.float32)
        M = pred_R # gt_R

        # for i in range(3):
        #     sym = axis_symmetry[i]
        #     if sym == 1:
        #         # M[i] *= -1
        #         rotation = np.identity(3) * -1
        #         ri = (i+1)%3
        #         rotation[ri,ri] = 1
        #         M = np.dot(rotation, M)
        final_pose[:4] = mat2quat(M)
        final_pose[4:] = centroid

        factor_depth = img_data['factor_depth']
        meta_data = {'intrinsic_matrix': intrinsic_matrix.tolist(), 'factor_depth': factor_depth}
        render_object_pose(img, depth, [cls], meta_data, [final_pose], points)

    # metric_mean = np.mean(metric_list, axis=0)
    # print("Mean ADD: %.3f, ADD 2: %.3f"%(metric_mean[0], metric_mean[1]))
    return metric_list


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

    dataset_dir = "./datasets/FAT"
    root_dir = osp.join(dataset_dir, "data")
    ann_file = osp.join(dataset_dir, "coco_fat_train_3500.json")

    data_loader = FATDataLoader3(root_dir, ann_file)
    # data = data_loader.next_batch(8)
    # data_loader.visualize(data)
    # data_loader.convert_data_batch_to_tensor(data)

    C = num_classes*3*3
    model = ConvNet(in_channels=3, out_channels=C)
    model.cuda()

    save_path = "model_vf_0.pth"
    model.load_state_dict(torch.load(save_path))
    print("Loaded %s"%(save_path))

    n_iters = 3000
    lr = 1e-4
    # train(model, data_loader, n_iters=n_iters, lr=lr)
    # torch.save(model.state_dict(), save_path)

    test_ann_file = osp.join(dataset_dir, "coco_fat_test_500.json")
    # test_ann_file = osp.join(dataset_dir, "coco_fat_mixed_temple_1_n100.json")
    test_data_loader = FATDataLoader3(root_dir, test_ann_file, shuffle=False)
    results = test(model, test_data_loader, batch_sz=100, visualize=True)

    for ix,cls in enumerate(test_data_loader._classes):
        if ix == 0:
            continue
        r = np.array(results[ix])
        print("(%s): Mean ADD: %.3f, ADD2: %.3f"%(cls, np.mean(r[:,0]), np.mean(r[:,1])))
