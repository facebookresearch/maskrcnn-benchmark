import json
import cv2
import numpy as np
import numpy.random as npr
import os.path as osp
import open3d

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class persistent_locals(object):
    def __init__(self, func):
        self._locals = {}
        self.func = func

    def __call__(self, *args, **kwargs):
        def tracer(frame, event, arg):
            if event=='return':
                l = frame.f_locals.copy()
                self._locals = l
                for k,v in l.items():
                    globals()[k] = v

        # tracer is activated on next call, return or exception
        sys.setprofile(tracer)
        try:
            # trace the function call
            res = self.func(*args, **kwargs)
            
        finally:
            # disable tracer and replace with old one
            sys.setprofile(None)
        return res

    def clear_locals(self):
        self._locals = {}

    @property
    def locals(self):
        return self._locals


def normalize(x, xmin=None, xmax=None):
    xmin = np.min(x) if xmin is None else xmin
    xmax = np.max(x) if xmax is None else xmax
    nx = x - xmin
    nx /= (xmax - xmin)
    return nx

def backproject_camera(im_depth, intrinsics, factor_depth=1):

    depth = im_depth.astype(np.float32, copy=True) / factor_depth

    # get intrinsic matrix
    K = intrinsics.copy()
    K = np.matrix(K)
    K = np.reshape(K,(3,3))
    Kinv = np.linalg.inv(K)

    # compute the 3D points        
    width = depth.shape[1]
    height = depth.shape[0]

    # construct the 2D points matrix
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    x2d = np.stack((x, y, ones), axis=2).reshape(width*height, 3)

    # backprojection
    R = Kinv * x2d.transpose()

    # compute the 3D points
    X = np.multiply(np.tile(depth.reshape(1, width*height), (3, 1)), R)

    return np.array(X)


def create_cloud(points, normals=[], colors=[], T=None):
    cloud = open3d.PointCloud()
    cloud.points = open3d.Vector3dVector(points)
    if len(normals) > 0:
        assert len(normals) == len(points)
        cloud.normals = open3d.Vector3dVector(normals)
    if len(colors) > 0:
        assert len(colors) == len(points)
        cloud.colors = open3d.Vector3dVector(colors)

    if T is not None:
        cloud.transform(T)
    return cloud

def render_predicted_depths(im, depth, pred_depths, intrinsics, factor_depth=1):
    rgb = im.copy()
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32)[:,:,::-1] / 255

    X = backproject_camera(depth, intrinsics, factor_depth)
    cloud_rgb = rgb # .astype(np.float32)[:,:,::-1] / 255
    cloud_rgb = cloud_rgb.reshape((cloud_rgb.shape[0]*cloud_rgb.shape[1],3))
    scene_cloud = create_cloud(X.T, colors=cloud_rgb)

    # for dp in pred_depths:
    X = backproject_camera(pred_depths, intrinsics, factor_depth)
    pred_cloud = create_cloud(X.T)
    open3d.draw_geometries([scene_cloud, pred_cloud])


def show_depth_cloud(depth, img, intrinsics, factor_depth=1):
    rgb = img.copy()
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32)[:,:,::-1] / 255

    X = backproject_camera(depth, intrinsics, factor_depth)
    rgb = rgb.reshape((rgb.shape[0]*rgb.shape[1],3))
    scene_cloud = create_cloud(X.T, colors=rgb)
    open3d.draw_geometries([scene_cloud])

def show_depth_cloud2(depth, img, intrinsics, factor_depth=1):
    rgb = img.copy()
    X = backproject_camera(depth, intrinsics, factor_depth)
    scene_cloud = create_cloud(X.T, colors=rgb)
    open3d.draw_geometries([scene_cloud])

def get_seg_mask_from_coco_annot(annot, img_h, img_w):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    segs = np.array(annot['segmentation'])[0]  # [xyxy]
    segs = segs.reshape((int(len(segs)/2), 2)) # [[x,y],[x,y]]
    segs = segs.astype(np.int32)
    cv2.fillPoly(mask, [segs], 255)
    return mask

def get_cropped_img(img, bbox):
    # bbox: x,y,w,h
    return img[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]].copy()

def FT(x): return torch.FloatTensor(x)

global USE_SCALED

class FATDataLoader(object):
    def __init__(self):
        self.root = "../datasets/FAT/data"
        ann_file = "../datasets/FAT/coco_fat_debug.json"

        with open(ann_file, "r") as f:
            data = json.load(f)

        images = data['images']
        annots = data['annotations']

        self.img_index = dict((a['id'], ix) for ix,a in enumerate(images))
        self.total_annots = len(annots)
        self.images = images
        self.annots = annots
        self.data = data

    def next_batch(self, batch_sz):
        perm = npr.permutation(self.total_annots)[:batch_sz]
        annots = [self.annots[idx] for idx in perm]

        image_list = []
        mask_list = []
        depth_list = []
        bbox_list = []
        for ann in annots:
            img_id = ann['image_id']
            img_data = self.images[self.img_index[img_id]]

            # load img
            img_file = img_data["file_name"]
            img_file_path = osp.join(self.root, img_file)
            img = cv2.imread(img_file_path)
            if img is None:
                print("Could not read %s"%(img_file_path))

            # load depth
            depth_file = img_data['depth_file_name']
            depth_file_path = osp.join(self.root, depth_file)
            depth = cv2.imread(depth_file_path, cv2.IMREAD_UNCHANGED)
            if depth is None:
                print("Could not read %s"%(depth_file_path))

            factor_depth = img_data['factor_depth']
            depth = depth.astype(np.float32) / factor_depth

            # crop out annot from image
            img_h = img_data['height']
            img_w = img_data['width']
            mask = get_seg_mask_from_coco_annot(ann, img_data['height'], img_data['width'])
            bbox = ann['bbox']
            cropped_img = get_cropped_img(img, bbox)
            cropped_mask = get_cropped_img(mask, bbox)
            cropped_depth = get_cropped_img(depth, bbox)

            if USE_SCALED:
                # load meta
                meta = ann['meta']
                centroid = meta['centroid']
                bounds = meta['bounds']
                centroid_z = centroid[-1]
                min_z = bounds[0][-1]
                max_z = bounds[1][-1]

                cropped_depth = (normalize(cropped_depth, min_z, max_z) - 0.5) * 2  # (normalize to 0-1 then -0.5 to 0.5 then -1 to 1)
                # cropped_depth = norm_depth

            image_list.append(cropped_img)
            mask_list.append(cropped_mask)
            depth_list.append(cropped_depth)
            # bbox_list.append(bbox)

        return [image_list, mask_list, depth_list, annots]

    def convert_data_batch_to_tensor(self, data, resize_shape=56, use_cuda=False):
        sz = resize_shape
        image_list, mask_list, depth_list, _ = data
        N = len(image_list)
        t_image_list = np.zeros((N, sz, sz, 3), dtype=np.float32)
        t_mask_list = np.zeros((N, sz, sz), dtype=np.float32)
        t_depth_list = np.zeros((N, sz, sz), dtype=np.float32) 
        for ix, im in enumerate(image_list):
            t_im = cv2.resize(im, (sz, sz), interpolation=cv2.INTER_LINEAR)
            t_mask = cv2.resize(mask_list[ix], (sz, sz), interpolation=cv2.INTER_LINEAR)
            t_depth = cv2.resize(depth_list[ix], (sz, sz), interpolation=cv2.INTER_LINEAR)
            t_mask[t_mask > 0.5] = 1
            t_mask[t_mask <= 0.5] = 0

            t_image_list[ix] = t_im.astype(np.float32) / 255  # assumes uint8!
            t_mask_list[ix] = t_mask
            t_depth_list[ix] = t_depth

        t_image_list = np.transpose(t_image_list, [0,3,1,2])  # (N,H,W,3) to (N,3,H,W)
        t_image_tensor = FT(t_image_list)
        t_mask_tensor = FT(t_mask_list)
        t_depth_tensor = FT(t_depth_list)
        if use_cuda:
            t_image_tensor = t_image_tensor.cuda()
            t_mask_tensor = t_mask_tensor.cuda()
            t_depth_tensor = t_depth_tensor.cuda()

        return t_image_tensor, t_mask_tensor, t_depth_tensor


def l1_loss(x, y):
    return torch.abs(x - y)

def train(model, dg):

    epochs = 10
    n_iters = 1000
    batch_size = 4
    lr = 1e-3

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    losses = []
    for iter in range(n_iters):
        data = dg.next_batch(batch_size)

        image_tensor, mask_tensor, depth_tensor = dg.convert_data_batch_to_tensor(data, use_cuda=True)
        optimizer.zero_grad()

        output = model(image_tensor)
        output = output.squeeze()

        # loss
        loss_type = "l1"
        loss = l1_loss(output[mask_tensor==1], depth_tensor[mask_tensor==1]).mean()
        # loss_type = "mean_var"
        # loss = mean_var_loss(output[mask_tensor==1], depth_tensor[mask_tensor==1]).mean()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if iter % 20 == 0 and iter > 0:
            print("iter %d of %d -> Total loss (%s): %.3f"%(iter, n_iters, loss_type, np.mean(losses)))
            losses = []

    print("iter %d of %d -> Total loss: %.3f"%(iter, n_iters, loss.item()))

def paste_mask_on_image(mask, box, im_h, im_w, thresh=None, interp=cv2.INTER_LINEAR):
    w = box[2]# - box[0] + 1
    h = box[3]# - box[1] + 1
    w = max(w, 1)
    h = max(h, 1)

    resized = cv2.resize(mask, (w,h), interpolation=interp)

    if thresh is not None:#thresh >= 0:
        resized = resized > thresh

    x_0 = max(box[0], 0)
    x_1 = min(box[0] + box[2], im_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[1] + box[3], im_h)

    canvas = np.zeros((im_h, im_w), dtype=np.float32)
    canvas[y_0:y_1, x_0:x_1] = resized[(y_0 - box[1]) : (y_1 - box[1]), (x_0 - box[0]) : (x_1 - box[0])]
    return canvas

# def bbox_xywh_to_xyxy(bbox):
#     return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

# @persistent_locals
def test(model, dg, batch_sz = 8, verbose=False):
    model.eval()

    intrinsics = np.array([[768.1605834960938, 0.0, 480.0], [0.0, 768.1605834960938, 270.0], [0.0, 0.0, 1.0]])
    factor_depth = 10000
    data = dg.next_batch(batch_sz)
    image_list, mask_list, depth_list, annots = data

    # convert to tensors
    image_tensor, mask_tensor, depth_tensor = dg.convert_data_batch_to_tensor(data, use_cuda=True)

    preds = model(image_tensor)
    preds = preds.detach().cpu().numpy().squeeze()

    for ix,depth_pred in enumerate(preds):
        ann = annots[ix]
        img_id = ann['image_id']
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

        bbox = ann['bbox']
        bbox_w, bbox_h = bbox[-2:]
        height, width = img.shape[:2]

        # get mask
        mask = mask_tensor[ix].cpu().numpy()

        dp = depth_pred.copy()

        if USE_SCALED:
            # scale prediction to
            dp[dp >= 1] = 0
            dp[dp <= -1] = 0
            meta = ann['meta']
            centroid = meta['centroid']
            bounds = meta['bounds']
            centroid_z = centroid[-1]
            min_z = bounds[0][-1]
            max_z = bounds[1][-1]
            z_scale = max_z - min_z
            # randd = float(np.random.randint(90,110)) / 100
            # z_scale *= randd

            dp = centroid_z + dp * z_scale / 2

        # paste it on depth map
        dp[mask!=1] = 0
        dp = cv2.resize(dp, (bbox_w, bbox_h), interpolation=cv2.INTER_NEAREST)
        dp = paste_mask_on_image(dp, bbox, height, width, interp=cv2.INTER_LINEAR)

        render_predicted_depths(img, depth.astype(np.float32) / factor_depth, dp, intrinsics)

if __name__ == '__main__':

    USE_SCALED = True

    data_loader = FATDataLoader()
    # data = data_loader.next_batch(2)
    # image_list, mask_list, depth_list, annots = data
    # T_im, T_seg, T_depth = data_loader.convert_data_batch_to_tensor(data)

    # max_depth = 5
    # for img, mask, depth in zip(image_list, mask_list, depth_list):
    #     cv2.imshow("depth", normalize(depth, 0, max_depth))
    #     cv2.imshow("mask", mask)
    #     cv2.imshow("img", img)
    #     cv2.waitKey(0)

    if USE_SCALED:
        save_path = "./model_0.pth"
    else:
        save_path = "./model_noscale_0.pth"

    from conv_test import ConvNet
    model = ConvNet(in_channels=3)
    model.cuda()

    model.load_state_dict(torch.load(save_path))

    # train(model, data_loader)
    # torch.save(model.state_dict(), save_path)

    test(model, data_loader, batch_sz=8)
