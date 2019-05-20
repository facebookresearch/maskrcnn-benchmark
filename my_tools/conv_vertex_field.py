"""
Fully conv on keypoint vertex field
"""

import cv2
import numpy as np
import numpy.random as npr
import os
import os.path as osp
import copy
from collections import defaultdict
# import open3d
# from transforms3d.quaternions import quat2mat, mat2quat

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from conv_depth import FATDataLoader, get_cropped_img, get_seg_mask_from_coco_annot, \
        FT, LT, conv_transpose2d_by_factor, get_random_color#, create_cloud#, backproject_camera,
# from conv_test import get_4x4_transform, average_point_distance_metric
# from conv_pose import render_object_pose

MAX_KEYPOINTS = 9
MAX_CHANNELS = MAX_KEYPOINTS * 2  # x,y per keypoint

RESIZE_SHAPE = 96

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

def normalize(x, xmin=None, xmax=None):
    xmin = np.min(x) if xmin is None else xmin
    xmax = np.max(x) if xmax is None else xmax
    nx = x - xmin
    if (xmax - xmin) > 0:
        nx /= (xmax - xmin)
    return nx

def generate_vertex_center_mask(label_mask, center):
    # c = np.zeros((2, 1), dtype=np.float32)
    # for ind, cls in enumerate(cls_i):
    c = np.expand_dims(center, axis=1) 
    h,w = label_mask.shape
    vertex_centers = np.zeros((h, w, 2), dtype=np.float32)  # channels first, as in pytorch convention
    # z = pose[2, 3]
    y, x = np.where(label_mask != 0)

    R = c - np.vstack((x, y))
    # compute the norm
    N = np.linalg.norm(R, axis=0) + 1e-10
    # normalization
    R = R / N # np.divide(R, np.tile(N, (2,1)))
    # assignment
    vertex_centers[y, x, 0] = R[0, :]
    vertex_centers[y, x, 1] = R[1, :]
    # if depth is not None:
    #     assert depth.shape == (h, w)
    #     vertex_centers[2, y, x] = depth[y, x]
    return vertex_centers

def resize_tensor(t, H, W, mode='bilinear'):
    return F.upsample(t, size=(H, W), mode=mode)


def draw_vert_field_canvas(vert_fields):
    """
    :param vert_fields: shape (C,H,W,2)
    :return: shape (H * C, W * 2)
    """
    total = len(vert_fields)
    if total == 0:
        return None
    h, w = vert_fields[0].shape[:2]
    canvas_vf = np.zeros((h * total, w * 2), dtype=np.float32)

    for jx, vf in enumerate(vert_fields):  # iterate each vert field
        cx = normalize(vf[:, :, 0], -1, 1)
        cy = normalize(vf[:, :, 1], -1, 1)
        canvas_vf[jx * h:(jx + 1) * h, :w] = cx
        canvas_vf[jx * h:(jx + 1) * h, w:] = cy

    return canvas_vf

class DataLoader(FATDataLoader):
    def __init__(self, root_dir, ann_file, shuffle=True):
        super(DataLoader, self).__init__(root_dir, ann_file, shuffle=shuffle)

    def next_batch(self, batch_sz, max_pad=0, random_crop=False):
        perm = self.get_next_batch_perm(batch_sz)
        annots = [self.annots[idx] for idx in perm]

        image_list = []  # [(H,W,3)]
        mask_list = [] # [(H,W)]
        bbox_list = []
        labels_list = []     
        vertex_field_list = [] # [(H,W,MAX_KEYPOINTS,2)]
        keypoints_list = []
        for ann in annots:
            img_id = ann['image_id']
            cls = ann['category_id']
            img_data = self.images[self.img_index[img_id]]

            # print(img_id)

            # load img
            img_file = img_data["file_name"]
            img_file_path = osp.join(self.root, img_file)
            img = cv2.imread(img_file_path)
            if img is None:
                print("Could not read %s"%(img_file_path))

            img_h = img_data['height']
            img_w = img_data['width']

            # get bbox and perform random crop
            x1,y1,bw,bh = ann['bbox']
            x2 = x1 + bw
            y2 = y1 + bh

            if max_pad > 0:
                min_x = max(0, x1 - max_pad)
                min_y = max(0, y1 - max_pad)
                max_x = min(img_w, x2 + max_pad)
                max_y = min(img_h, y2 + max_pad)
                if random_crop:
                    # try:
                    x1 = npr.randint(min_x, x1) if min_x < x1 else x1
                    y1 = npr.randint(min_y, y1) if min_y < y1 else y1
                    x2 = npr.randint(x2, max_x) if max_x > x2 else x2
                    y2 = npr.randint(y2, max_y) if max_y > y2 else y2
                    # except ValueError:
                    #     print(x1,y1,x2,y2)
                    #     print(min_x,min_y,max_x,max_y)
                else:
                    x1 = min_x
                    x2 = max_x
                    y1 = min_y
                    y2 = max_y

            bbox = [x1,y1,x2-x1,y2-y1]  # x1 y1 w h

            # crop out annot from image
            mask = get_seg_mask_from_coco_annot(ann, img_h, img_w)
            cropped_img = get_cropped_img(img, bbox)
            cropped_mask = get_cropped_img(mask, bbox)

            # get meta data
            meta = ann['meta']
            keypoints = np.array(meta['keypoints'][:MAX_KEYPOINTS])
            # minus by bbox
            keypoints[:,0] -= bbox[0]
            keypoints[:,1] -= bbox[1]

            # vert_field =
            mh, mw = cropped_mask.shape[:2]
            cropped_vert_field = np.zeros((MAX_KEYPOINTS, mh, mw, 2), dtype=np.float32)
            for jx, kp in enumerate(keypoints):
                cropped_vert_field[jx] = generate_vertex_center_mask(cropped_mask, kp)  # (H,W,2)

            image_list.append(cropped_img)
            mask_list.append(cropped_mask)
            vertex_field_list.append(cropped_vert_field)
            labels_list.append(cls)
            bbox_list.append(bbox)
            keypoints_list.append(keypoints)

        return [image_list, labels_list, bbox_list, mask_list, vertex_field_list, keypoints_list, annots]

    def convert_data_batch_to_tensor(self, data, resize_shape=56, use_cuda=False):
        sz = resize_shape
        image_list, labels_list, bbox_list, mask_list, vertex_field_list, keypoints_list, _ = data
        
        N = len(image_list)
        t_image_list = np.zeros((N, sz, sz, 3), dtype=np.float32)
        t_mask_list = np.zeros((N, sz, sz), dtype=np.float32)
        t_vert_field_list = np.zeros((N, sz, sz, MAX_KEYPOINTS, 2), dtype=np.float32) 

        for ix, im in enumerate(image_list):
            ori_h, ori_w = im.shape[:2]
            t_im = cv2.resize(im, (sz, sz), interpolation=cv2.INTER_LINEAR)
            mask = mask_list[ix]
            t_mask = cv2.resize(mask, (sz, sz), interpolation=cv2.INTER_LINEAR)

            t_im = t_im.astype(np.float32) / 255  # assumes 0-255!
            t_mask[t_mask > 0.5] = 1
            t_mask[t_mask <= 0.5] = 0

            # generate keypoint vertex fields
            keypoints = keypoints_list[ix].astype(np.float32)
            keypoints[:, 0] *= sz / ori_w # x
            keypoints[:, 1] *= sz / ori_h # y
            t_vf = np.zeros((MAX_KEYPOINTS, sz, sz, 2), dtype=np.float32)  # (MAX_KEYPOINTS, sz, sz, 2)
            for jx, kp in enumerate(keypoints):
                t_vf[jx] = generate_vertex_center_mask(t_mask, kp)

            # DON'T RESIZE DUE TO INTERPOLATION
            # vf = np.transpose(vertex_field_list[ix], [1,2,0,3])  # (MAX_KEYPOINTS,h,w,2) to (h,w,MAX_KEYPOINTS,2)
            # h,w = vf.shape[:2]
            # vf = vf.reshape((h,w,MAX_KEYPOINTS*2))  # (h,w,MAX_KEYPOINTS*2)
            # t_vf = cv2.resize(vf, (sz, sz), interpolation=cv2.INTER_LINEAR)  # (sz,sz,MAX_KEYPOINTS*2)
            # t_vf = t_vf.reshape((sz,sz,MAX_KEYPOINTS,2))

            # t_im = (t_im - 0.5) * 2
            t_image_list[ix] = t_im
            t_mask_list[ix] = t_mask
            t_vert_field_list[ix] = np.transpose(t_vf, [1,2,0,3])  # (sz, sz, MAX_KEYPOINTS, 2)

        t_image_list = np.transpose(t_image_list, [0,3,1,2])  # (N,H,W,3) to (N,3,H,W)
        t_image_tensor = FT(t_image_list)
        t_mask_tensor = FT(t_mask_list)
        t_vf_tensor = FT(t_vert_field_list)
        t_labels_tensor = LT(labels_list)
        if use_cuda:
            t_image_tensor = t_image_tensor.cuda()
            t_mask_tensor = t_mask_tensor.cuda()
            t_vf_tensor = t_vf_tensor.cuda()
            t_labels_tensor = t_labels_tensor.cuda()

        return t_image_tensor, t_labels_tensor, t_mask_tensor, t_vf_tensor

    def visualize(self, data):
        image_list, labels_list, mask_list, vertex_field_list, keypoints_list, annots = data

        for ix,ann in enumerate(annots):
            img = image_list[ix]
            mask = mask_list[ix]
            vfs = vertex_field_list[ix]  # (MAX_KEYPOINTS,H,W,2)
            # h,w = mask.shape[:2]
            # meta = ann['meta']
            # pose = meta['pose']  # qw,qx,qy,qz,x,y,z

            canvas_vf = draw_vert_field_canvas(vfs)

            cv2.imshow("vertex_kp", canvas_vf)
            cv2.imshow("im", img)
            cv2.imshow("mask", mask)
            cv2.waitKey(0)

def line(p1, p2):
    N = len(p1)
    assert len(p1) == len(p2)
    x = np.zeros((N,3))
    x[:,0] = p1[:,1] - p2[:,1]
    x[:,1] = p2[:,0] - p1[:,0]
    x[:,2] = -(p1[:,0]*p2[:,1] - p2[:,0]*p1[:,1])
    return x

def compute_intersections(pt1, pt2, vector1, vector2):
    N = len(pt1)
    assert len(pt1) == len(pt2)

    TO_REMOVE = 1e-8

    L1 = line(pt1, pt1 + vector1)
    L2 = line(pt2, pt2 + vector2)

    d = np.zeros((N,3))
    D  = L1[:,0] * L2[:,1] - L1[:,1] * L2[:,0] + TO_REMOVE
    Dx = L1[:,2] * L2[:,1] - L1[:,1] * L2[:,2]
    Dy = L1[:,0] * L2[:,2] - L1[:,2] * L2[:,0]
    d[:,0] = Dx / D
    d[:,1] = Dy / D
    d[:,2] = D != TO_REMOVE
    return d

class ConvNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(ConvNet, self).__init__()

        conv1_filters = 64
        conv2_filters = 128
        conv3_filters = 512

        self.conv1 = nn.Conv2d(in_channels, conv1_filters, kernel_size=5, stride=2, padding=5//2)  # stride 2
        self.conv2 = nn.Conv2d(conv1_filters, conv2_filters, kernel_size=5, stride=2, padding=5//2)  # stride 4
        self.conv3 = nn.Conv2d(conv2_filters, conv3_filters, kernel_size=3, stride=2, padding=3//2)  # stride 8
        self.bn1 = nn.BatchNorm2d(conv1_filters)
        self.bn2 = nn.BatchNorm2d(conv2_filters)
        self.bn3 = nn.BatchNorm2d(conv3_filters)

        conv_t_filters = 128
        conv_t2_filters = conv_t_filters * 2
        self.conv_t1 = conv_transpose2d_by_factor(conv3_filters, conv2_filters, factor=2)  # stride 4
        self.conv_t2 = conv_transpose2d_by_factor(conv2_filters, conv1_filters, factor=2)  # stride 2
        self.conv_t3 = conv_transpose2d_by_factor(conv1_filters, conv_t_filters, factor=2)  # stride 1
        self.reg = nn.Conv2d(conv_t_filters, conv_t2_filters, kernel_size=5, stride=1, padding=5 // 2)
        self.reg2 = nn.Conv2d(conv_t2_filters, out_channels, kernel_size=5, stride=1, padding=5 // 2)

    def forward(self, x):
        # batch_sz = len(x)
        c1 = F.relu(self.bn1(self.conv1(x)))
        c2 = F.relu(self.bn2(self.conv2(c1)))
        c3 = F.relu(self.bn3(self.conv3(c2)))
        # c1 = F.relu(self.conv1(x))
        # c2 = F.relu(self.conv2(c1))
        # c3 = F.relu(self.conv3(c2))
        ct1 = F.relu(self.conv_t1(c3))
        ct1 = torch.add(ct1, c2)
        ct2 = F.relu(self.conv_t2(ct1))
        ct2 = torch.add(ct2, c1)
        ct3 = F.relu(self.conv_t3(ct2))
        out = F.relu(self.reg(ct3))
        out = self.reg2(out)
        return out

def train(model, dg, n_iters=1000, lr=1e-3):

    from tensorboardX import SummaryWriter
    writer = SummaryWriter()

    batch_size = 32

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    loss_type = "L1"

    all_losses = []
    losses = []
    for iter in range(n_iters):
        data = dg.next_batch(batch_size, max_pad=50, random_crop=True)

        # normalize the pixels by pixel mean
        image_list = data[0]
        for ix,im in enumerate(image_list):
            image_list[ix] = im.astype(np.float32) - PIXEL_MEAN

        image_tensor, labels_tensor, mask_tensor, vert_field_tensor = \
                dg.convert_data_batch_to_tensor(data, use_cuda=True, resize_shape=RESIZE_SHAPE)
        optimizer.zero_grad()

        vf_pred = model(image_tensor)

        if vert_field_tensor.numel() == 0 or vf_pred.numel() == 0:
            continue

        vf_pred = torch.tanh(vf_pred)

        N, C, H, W = vf_pred.shape  # N,classes*MAX_CHANNELS,H,W
        vf_pred = vf_pred.reshape((N,C//MAX_CHANNELS,MAX_CHANNELS,H,W))
        pos_inds = torch.arange(N)
        vf = vf_pred[pos_inds, labels_tensor]  # N,MAX_CHANNELS,H,W

        target = vert_field_tensor.reshape((N,H,W,MAX_CHANNELS))  # N,H,W,MAX_CHANNELS//2,2 to N,H,W,MAX_CHANNELS
        pp = vf.permute(0,2,3,1).clone()  # N,MAX_CHANNELS,H,W to N,H,W,MAX_CHANNELS

        # loss
        abs_diff = torch.abs(pp - target)
        loss = smooth_l1(abs_diff[mask_tensor==1])
        loss.backward()
        optimizer.step()
        loss_value = loss.item()

        losses.append(loss_value)
        all_losses.append(loss_value)

        writer.add_scalar('data/kp_vert_field_l1_loss', loss_value, iter)

        if iter % 20 == 0 and iter > 0:
            print("iter %d of %d -> (%s) Total loss: %.4f, Avg loss: %.4f"%(iter, n_iters, loss_type, np.mean(losses), np.mean(all_losses)))
            losses = []

    print("iter %d of %d -> Total loss: %.4f, Avg loss: %.4f"%(iter, n_iters, np.mean(losses), np.mean(all_losses)))
    writer.close()


def get_voting_mean_pixel(results, top_k=20):
    top_pixels = results[:,:2][:top_k]
    sorted_n = results[:,2][:top_k]

    total = np.sum(sorted_n)
    weights = sorted_n.astype(np.float32) / total
    
    # estimate mean and draw
    weighted_pixels = top_pixels * weights[:,None]
    mean_pixel = np.sum(weighted_pixels, axis=0)
    return mean_pixel

def draw_voting_results(img, results, top_k=20):
    h, w = img.shape[:2]
    # sz = h * w
    
    img_mean_px = img.copy()
    img_heatmap = img.copy()

    mean_pixel = get_voting_mean_pixel(results, top_k)
    cv2.circle(img_mean_px, tuple(np.round(mean_pixel).astype(np.int32)), 2, RED, -1)

    # draw heatmap of top k points
    top_pixels = results[:,:2][:top_k]
    sorted_n = results[:,2][:top_k]
    if len(sorted_n) == 0:
        return img_mean_px, img_heatmap

    top_n = sorted_n[0]

    valid_x = np.logical_and(top_pixels[:, 0] >= 0, top_pixels[:, 0] < w)
    valid_y = np.logical_and(top_pixels[:, 1] >= 0, top_pixels[:, 1] < h)
    valid_n = np.logical_and(valid_x, valid_y)
    tot = np.sum(valid_n)
    # print(tot)

    if tot > 0:
        vn = sorted_n[valid_n]
        valid_px = top_pixels[valid_n]
        norm_n = normalize(vn.astype(np.float32), 0, top_n)
        heat = cv2.applyColorMap((norm_n * 255).astype(np.uint8), cv2.COLORMAP_JET).squeeze()
        heatmap = np.zeros(img_heatmap.shape, dtype=np.uint8)

        heatmap[valid_px[:,1], valid_px[:,0]] = heat#[valid_n]
        alpha = 0.7
        cv2.addWeighted(heatmap, alpha, img_heatmap, 1 - alpha, 0, img_heatmap)

    return img_mean_px, img_heatmap

def hough_vote_ransac2(mask, vertex_centers, N=10000, min_vote=2):
    """
    mask: (H,W)
    vertex_centers: (H,W,2)  # last axis for x,y normalized vectors
    """
    h,w = mask.shape
    sz = h * w
    mask_y, mask_x = np.where(mask!=0)

    mask_pixels = np.hstack((mask_x[:,None], mask_y[:,None]))
    total = len(mask_pixels)

    r_mp = np.random.randint(0,total,size=(2,N))
    sample_mp = mask_pixels[r_mp[0]]  # N,2 -> N,(x,y)
    sample_mp2 = mask_pixels[r_mp[1]]  # N,2 -> N,(x,y)
    sample_verts = vertex_centers[sample_mp[:,1],sample_mp[:,0]]  # N,2 -> N,(u,v)
    sample_verts2 = vertex_centers[sample_mp2[:,1],sample_mp2[:,0]]  # N,2 -> N,(u,v)
    intersects = compute_intersections(sample_mp, sample_mp2, sample_verts, sample_verts2)
    valid = intersects[:,-1] == 1

    # mp = sample_mp[valid]
    # mp2 = sample_mp2[valid]
    valid_intersects = np.round(intersects[valid]).astype(np.int32)  # N,2 -> N,(x,y)

    # # TODO: YOU CAN'T JUST ADD THE HEIGHT AND WIDTH VALUES. WHAT IF HEIGHT/WIDTH IS NEGATIVE?
    # flattened = valid_intersects[:,1] * w + valid_intersects[:,0]  # N
    # f,votes=np.unique(flattened, return_counts=True)

    votes = defaultdict(int)
    for v_intersect in valid_intersects:
        key = tuple(v_intersect[:2])  # (x,y)
        votes[key] += 1

    pixels = np.array(list(votes.keys()))  # (N, 2)
    votes = np.array(list(votes.values())) # (N)
    
    # filter out votes below min vote threshold
    valid_idx = votes >= min_vote 
    valid_votes = votes[valid_idx]
    valid_pixels = pixels[valid_idx]

    # sort the votes from largest to smallest
    sorted_idx = np.argsort(valid_votes)[::-1] 

    sorted_n = valid_votes[sorted_idx]
    sorted_pixels = valid_pixels[sorted_idx]

    result = np.zeros((len(sorted_n), 3), dtype=np.int32) # x,y,votes
    result[:,0:2] = sorted_pixels
    result[:,2] = sorted_n # votes

    return result

def test(model, dg, batch_sz=8):
    import open3d

    model.eval()

    # TODO: REMOVE
    point_keypoints = np.array([
        [0.04965743, 0.0696693,  0.00412764],
        [-0.04912861,  0.06956633, -0.0067097 ],
        [-0.04532503, -0.06941872, -0.02012832],
        [ 0.05041973, -0.069449,    0.00409536],
        [ 0.01400065, -0.06746875, -0.04612001],
        [-0.01939844, -0.06946825,  0.0465417 ],
        [ 0.00469201,  0.06574368, -0.05085601],
        [-0.00595804,  0.0699745,   0.04865649],
        [0,0,0]
    ])
    point_cloud = open3d.read_point_cloud("/home/bot/hd/datasets/FAT/models/002_master_chef_can/002_master_chef_can.pcd")
    points = np.asarray(point_cloud.points)
    # TODO

    data = dg.next_batch(batch_sz, max_pad=5, random_crop=False)
    image_list, labels_list, bbox_list, mask_list, vertex_field_list, keypoints_list, annots = data

    # normalize the pixels by pixel mean
    image_list = data[0]
    ori_image_list = copy.copy(image_list)
    for ix, im in enumerate(image_list):
        image_list[ix] = im.astype(np.float32) - PIXEL_MEAN

    image_tensor, labels_tensor, mask_tensor, vert_field_tensor = \
        dg.convert_data_batch_to_tensor(data, use_cuda=True, resize_shape=RESIZE_SHAPE)

    N_RANSAC = 10000
    min_votes = 4
    N = N_RANSAC

    preds = model(image_tensor)
    preds = torch.tanh(preds).detach().cpu().numpy()  # .squeeze()

    mask_tensor_cpu = mask_tensor.cpu().numpy()
    vert_field_tensor_cpu = vert_field_tensor.cpu().numpy()

    results = []
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    pnp_algorithm = cv2.SOLVEPNP_ITERATIVE

    for ix, vf_pred in enumerate(preds):
        img = ori_image_list[ix]

        ori_h, ori_w = img.shape[:2]

        # resize image to batch size shape
        img = cv2.resize(img, (RESIZE_SHAPE, RESIZE_SHAPE))

        cls = labels_list[ix]
        mask = mask_tensor_cpu[ix]
        ann = annots[ix]

        # pred
        # img2 = img.copy()
        C, H, W = vf_pred.shape
        vf_pred = vf_pred.reshape((C // MAX_CHANNELS, MAX_CHANNELS, H, W))

        vf_pred = vf_pred[cls]  # (MAX_CHANNELS, H, W)
        pred = np.transpose(vf_pred, [1,2,0])  # (H, W, MAX_CHANNELS)
        pred[mask!=1] = 0

        pred = pred.reshape((H,W,MAX_CHANNELS//2,2))  # (H, W, C, 2)
        gt = vert_field_tensor_cpu[ix]

        pred_t = np.transpose(pred, [2, 0, 1, 3])  # (C,H,W,2)
        gt_t = np.transpose(gt, [2, 0, 1, 3])  # (C,H,W,2)
        
        # draw gt and pred on canvas
        canvas_pred_vf = draw_vert_field_canvas(pred_t) 
        canvas_gt_vf = draw_vert_field_canvas(gt_t)

        # # get meta data
        # bbox = ann['bbox']
        # meta = ann['meta']
        # keypoints = np.array(meta['keypoints'][:MAX_KEYPOINTS])
        # # minus by bbox
        # keypoints[:, 0] -= bbox[0]
        # keypoints[:, 1] -= bbox[1]
        keypoints = keypoints_list[ix]

        img_copy = img.copy()

        for jx, kp in enumerate(keypoints):
            pt = np.array([kp[0] / ori_w * W, kp[1] / ori_h * H]).astype(np.int32)
            # pt = kp
            cv2.circle(img_copy, tuple(pt), 2, GREEN, -1)

        C = MAX_KEYPOINTS
        canvas_heatmap_pred = np.zeros((C * H, W*3, 3), dtype=np.uint8)
        # canvas_heatmap_gt = canvas_heatmap_pred.copy()

        # run ransac hough voting and draw voting results
        results_pred = []
        results_gt = []
        for jx in range(C):
            y1, y2 = jx * W, (jx+1) * W
            
            result_pred = hough_vote_ransac2(mask, pred_t[jx], N=N, min_vote=min_votes) # N,3 -> (pixel x, pixel y, votes)
            result_gt = hough_vote_ransac2(mask, gt_t[jx], N=N, min_vote=min_votes)  # N,3 -> (pixel x, pixel y, votes)
            mean_pred_px, heatmap_pred = draw_voting_results(img, result_pred, top_k=100)
            mean_gt_px, heatmap_gt = draw_voting_results(img, result_gt, top_k=100)

            canvas_heatmap_pred[y1:y2, :W] = heatmap_pred
            canvas_heatmap_pred[y1:y2, W:W*2] = mean_pred_px
            canvas_heatmap_pred[y1:y2, W*2:] = mean_gt_px
            # canvas_heatmap_gt[y1:y2, :W] = heatmap_gt
            # canvas_heatmap_gt[y1:y2, W:] = mean_gt_px

            results_pred.append(result_pred)
            results_gt.append(result_gt)

        out_results_file = "results_pred.npy"
        np.save(out_results_file, results_pred)
        print("Saved to %s"%(out_results_file))

        # show full image and bbox
        ann = annots[ix]
        img_id = ann['image_id']
        img_data = dg.images[dg.img_index[img_id]]

        # load img
        img_file = img_data["file_name"]
        
        img_file_path = osp.join(dg.root, img_file)
        full_img = cv2.imread(img_file_path)
        if full_img is None:
            print("Could not read %s" % (img_file_path))
        else:
            x1,y1,bw,bh = bbox_list[ix]
            x2 = x1 + bw + 1
            y2 = y1 + bh + 1
            full_img = cv2.rectangle(full_img, (x1,y1), (x2,y2), RED, 2)
            cv2.imshow("bbox", full_img)

            # load intrinsic matrix
            intrinsic_matrix = np.array(img_data["meta"]["intrinsic_matrix"])

            # get gt pose
            pose = np.array(ann['meta']["pose"])  # 4x4 matrix
            # print(img_file_path)
            # print(pose)
            
            # get 2d projected cuboid
            full_img2 = full_img.copy()
            project_points(points, pose[:3,:3], pose[:3,3], intrinsic_matrix, img=full_img2)
            cv2.imshow("gt_proj_2d", full_img2)

            # get projected keypoints and run PnP
            proj_gt_keypoints = np.array([get_voting_mean_pixel(results_gt[jx], top_k=20) for jx in range(C)])
            proj_pred_keypoints = np.array([get_voting_mean_pixel(results_pred[jx], top_k=20) for jx in range(C)])
            # set 2d values back to original image 
            proj_gt_keypoints = proj_gt_keypoints * np.array([ori_w / W, ori_h / H]) + np.array([x1, y1])
            proj_pred_keypoints = proj_pred_keypoints * np.array([ori_w / W, ori_h / H]) + np.array([x1, y1])

            print("x1,y1 = (%d,%d); ori_h,ori_w = (%d,%d); H,W = (%d,%d)"%(x1,y1,ori_h,ori_w,H,W))
            
            # solve pnp n draw 2d projected pts 
            success, M = solve_pnp(point_keypoints, proj_gt_keypoints, intrinsic_matrix, flags=pnp_algorithm)
            full_img3 = full_img.copy()
            project_points(points, M[:3,:3], M[:3,3], intrinsic_matrix, img=full_img3)
            cv2.imshow("gt_proj_2d_vote_pnp", full_img3)

            # solve pnp n draw 2d projected pts 
            success, M = solve_pnp(point_keypoints, proj_pred_keypoints, intrinsic_matrix, flags=pnp_algorithm)
            full_img4 = full_img.copy()
            project_points(points, M[:3,:3], M[:3,3], intrinsic_matrix, img=full_img4)
            cv2.imshow("pred_proj_2d_vote_pnp", full_img4)

        # vis
        cv2.imshow("pred", canvas_pred_vf)
        cv2.imshow("gt", canvas_gt_vf)
        cv2.imshow("keypoints", img_copy)
        # cv2.imshow("kp", img)
        cv2.imshow("pred_heat, mean pred, mean gt", canvas_heatmap_pred)
        # cv2.imshow("gt_heat", canvas_heatmap_gt)
        cv2.waitKey(0)

def solve_pnp(pts_3d, pts_2d, intrinsic_matrix, dist_coeffs=np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE):
    success, rvec, tvec = cv2.solvePnP(pts_3d, pts_2d, intrinsic_matrix, dist_coeffs, flags=flags)
    M = np.eye(4)
    if success:
        tvec = tvec.squeeze()
        R, j = cv2.Rodrigues(rvec)
        M[:3,:3] = R
        M[:3,3] = tvec
    return success, M

def project_points(points, R, t, intrinsic_matrix, dist_coeffs=np.zeros((4,1)), img=None):
    pts_2d, _ = cv2.projectPoints(points, R, t, intrinsic_matrix, dist_coeffs)
    pts_2d = pts_2d.squeeze()

    if img is not None:
        assert len(img.shape) == 3
        pts_2d = pts_2d.astype(np.int32)
        for px in pts_2d:
            cv2.circle(img, tuple(px), 1, GREEN)
    return pts_2d

if __name__ == '__main__':
    CLASSES = ['__background__', '002_master_chef_can']

    num_classes = len(CLASSES)

    dataset_dir = "/home/bot/Documents/practice/render"
    root_dir = osp.join(dataset_dir, "out_train")
    ann_file = osp.join(dataset_dir, "coco_pyrender_train.json")

    data_loader = DataLoader(root_dir, ann_file)
    # data = data_loader.next_batch(8, max_pad=30, random_crop=True)
    # data_loader.visualize(data)
    # data_tensor = data_loader.convert_data_batch_to_tensor(data, resize_shape=RESIZE_SHAPE)

    model = ConvNet(in_channels=3, out_channels=num_classes*MAX_CHANNELS)
    model.cuda()
    # out = model.forward(data_tensor[0])

    save_path = "model_vert_field_0.pth"
    model.load_state_dict(torch.load(save_path))
    print("Loaded %s"%(save_path))

    n_iters = 500
    lr = 3e-4
    # train(model, data_loader, n_iters=n_iters, lr=lr)
    # torch.save(model.state_dict(), save_path)

    test_root_dir = osp.join(dataset_dir, "out_test")
    test_ann_file = osp.join(dataset_dir, "coco_pyrender_test.json")
    test_data_loader = DataLoader(test_root_dir, test_ann_file, shuffle=False)
    test(model, test_data_loader, batch_sz=50)
