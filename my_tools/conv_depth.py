"""
Fully conv regression on scaled depth OR raw depth
"""

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

from conv_test import conv_transpose2d_by_factor, create_cloud, backproject_camera

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

def render_predicted_depths(im, intrinsics, depth, pred_depths = [], pred_colors=[], factor_depth=1):
    rgb = im.copy()
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32)[:,:,::-1] / 255

    X = backproject_camera(depth, intrinsics, factor_depth)
    cloud_rgb = rgb # .astype(np.float32)[:,:,::-1] / 255
    cloud_rgb = cloud_rgb.reshape((cloud_rgb.shape[0]*cloud_rgb.shape[1],3))
    scene_cloud = create_cloud(X.T, colors=cloud_rgb)

    if not isinstance(pred_depths, list):
        pred_depths = [pred_depths]

    object_clouds = create_cloud([])
    if len(pred_depths) > 0:
        for i in range(len(pred_depths) - len(pred_colors)):
            pred_colors.append(get_random_color())
        for ix,pred_d in enumerate(pred_depths):
            X = backproject_camera(pred_d, intrinsics, factor_depth)
            color_arr = np.zeros((int(X.size / 3), 3), dtype=np.float32)
            color_arr[:] = pred_colors[ix]
            pred_cloud = create_cloud(X.T, colors=color_arr / 255)
            object_clouds += pred_cloud
    open3d.draw_geometries([scene_cloud, object_clouds])


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

def get_random_color():
    return (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))

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
def LT(x): return torch.LongTensor(x)

class FATDataLoader(object):
    def __init__(self, root_dir, ann_file, use_scaled_depth=False, shuffle=True):
        self.root = root_dir 
        self.ann_file = ann_file
        self.use_scaled_depth = use_scaled_depth

        with open(ann_file, "r") as f:
            data = json.load(f)

        images = data['images']
        annots = data['annotations']
        categories = data['categories']

        self._classes = ['__background'] + [c['name'] for c in categories]  # +1 for bg class
        self.num_classes = len(self._classes)

        self.img_index = dict((a['id'], ix) for ix,a in enumerate(images))
        self.total_annots = len(annots)
        self.images = images
        self.annots = annots
        self.data = data

        self.shuffle = shuffle
        self.cur_idx = 0
        self.permutation = None
        self._reset_permutation()

        self.SYMMETRIES_DICT = {
            '__background__': [0,0,0],
            '002_master_chef_can': [0,0,0],
            '003_cracker_box': [0,0,0],
            '004_sugar_box': [0,0,0],
            '005_tomato_soup_can': [0,0,0],
            '006_mustard_bottle': [1,0,0],
            '007_tuna_fish_can': [0,1,0], # the green axis (top and bottom) looks similar 
            '008_pudding_box': [0,0,0],  # brown JELLO box
            '009_gelatin_box': [0,0,0],  # red JELLO box
            '010_potted_meat_can': [0,0,0],
            '011_banana': [0,0,0],
            '019_pitcher_base': [0,0,0],
            '021_bleach_cleanser': [0,0,0],
            '024_bowl': [1,0,1],
            '025_mug': [0,0,0],
            '035_power_drill': [0,0,0],
            '036_wood_block': [1,1,1],
            '037_scissors': [0,0,0],
            '040_large_marker': [1,0,1], # assume symmetric on the sides, since marker is very small to see the texture clearly
            '051_large_clamp': [0,1,1],
            '052_extra_large_clamp': [0,1,1],
            '061_foam_brick': [1,1,1]
        }
        self.SYMMETRIES = [v for k,v in self.SYMMETRIES_DICT.items()]


    def _reset_permutation(self):
        total = self.total_annots
        self.permutation = npr.permutation(total) if self.shuffle else np.arange(total)

    def next_batch(self, batch_sz):
        start_idx = self.cur_idx
        self.cur_idx += batch_sz
        perm = self.permutation[start_idx:self.cur_idx]
        if self.cur_idx >= self.total_annots:
            self._reset_permutation()
            self.cur_idx -= self.total_annots
            perm = np.hstack((perm, self.permutation[:self.cur_idx]))
        annots = [self.annots[idx] for idx in perm]

        image_list = []
        mask_list = []
        depth_list = []
        bbox_list = []
        labels_list = []
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

            if self.use_scaled_depth:
                # load meta
                meta = ann['meta']
                centroid = meta['centroid']
                bounds = meta['bounds']
                centroid_z = centroid[-1]
                min_z = bounds[0][-1]
                max_z = bounds[1][-1]

                cropped_depth = (normalize(cropped_depth, min_z, max_z) - 0.5) * 2  # (normalize to 0-1 then -0.5 to 0.5 then -1 to 1)
                cropped_depth[cropped_depth < -1] = 0
                cropped_depth[cropped_depth > 1] = 0
                # cropped_depth = norm_depth

            image_list.append(cropped_img)
            mask_list.append(cropped_mask)
            depth_list.append(cropped_depth)
            labels_list.append(cls)
            # bbox_list.append(bbox)

        return [image_list, labels_list, mask_list, depth_list, annots]

    def convert_data_batch_to_tensor(self, data, resize_shape=56, use_cuda=False):
        sz = resize_shape
        image_list, labels_list, mask_list, depth_list, _ = data
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

            t_im = t_im.astype(np.float32) / 255  # assumes 0-255!
            # t_im = (t_im - 0.5) * 2
            t_image_list[ix] = t_im
            t_mask_list[ix] = t_mask
            t_depth_list[ix] = t_depth

        t_image_list = np.transpose(t_image_list, [0,3,1,2])  # (N,H,W,3) to (N,3,H,W)
        t_image_tensor = FT(t_image_list)
        t_mask_tensor = FT(t_mask_list)
        t_depth_tensor = FT(t_depth_list)
        t_labels_tensor = LT(labels_list)
        if use_cuda:
            t_image_tensor = t_image_tensor.cuda()
            t_mask_tensor = t_mask_tensor.cuda()
            t_depth_tensor = t_depth_tensor.cuda()
            t_labels_tensor = t_labels_tensor.cuda()

        return t_image_tensor, t_labels_tensor, t_mask_tensor, t_depth_tensor


def l1_loss(x, y):
    return torch.abs(x - y)

def l2_loss(x, y):
    return (x - y) ** 2

def mean_var_loss(input, target, size_average=True):
    # mean_p = torch.mean(input)
    # mean_t = torch.mean(target)
    diff = target - input
    # mean_loss = torch.abs(mean_t - mean_p)
    md = torch.mean(diff)
    var_loss = (1 + torch.abs(diff - md)) ** 2 - 1
    # loss = 0.5 * torch.abs(diff) + 0.5 * var_loss
    loss = 0.5 * l1_loss(input, target) + 0.5 * var_loss
    return loss
    # if size_average:
    #     return loss.mean()
    # return loss.sum()

def berhu_loss(input, target, beta=0.2):#, size_average=True):
    abs_error = torch.abs(input - target)
    max_err = torch.max(abs_error)
    min_err = torch.min(abs_error)
    if max_err - min_err > 0.5:
        c = beta * torch.max(abs_error)
        cond = abs_error <= c
        loss = torch.where(cond, abs_error, (abs_error ** 2 + c ** 2) / (2 * c))
    else:
        loss = abs_error
    return loss


def train(model, dg, n_iters=1000, lr=1e-3):

    from tensorboardX import SummaryWriter
    writer = SummaryWriter()

    # epochs = 10
    # n_iters = 1000
    batch_size = 16

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    loss_type = "l1"
    loss_fn = l1_loss
    # loss_type = "berhu"
    # loss_fn = berhu_loss

    losses = []
    all_losses = []
    for iter in range(n_iters):
        data = dg.next_batch(batch_size)

        image_tensor, labels_tensor, mask_tensor, depth_tensor = dg.convert_data_batch_to_tensor(data, use_cuda=True)
        optimizer.zero_grad()

        outputs = model(image_tensor)
        N = outputs.size(0)
        channels = outputs.size(1)

        # loss
        loss = 0.0
        if channels == 1:  # class agnostic
            outputs = outputs.squeeze(1)
            mask = mask_tensor == 1
            loss = loss_fn(outputs[mask], depth_tensor[mask]).mean()
        else:
            pos_inds = torch.arange(N)
            outputs = outputs[pos_inds, labels_tensor]
            mask = mask_tensor == 1
            loss = loss_fn(outputs[mask], depth_tensor[mask]).mean()
            # for ix,output in enumerate(outputs):
            #     label = labels_tensor[ix]
            #     cls_output = output[label]
            #     mask = mask_tensor[ix] == 1
            #     loss += loss_fn(cls_output[mask], depth_tensor[ix][mask]).mean()
            # loss /= N

        loss.backward()
        optimizer.step()

        loss_value = loss.item()

        losses.append(loss_value)
        all_losses.append(loss_value)

        writer.add_scalar('data/loss', loss_value, iter)

        if iter % 20 == 0 and iter > 0:
            print("iter %d of %d -> (%s) Total loss: %.4f, Avg loss: %.4f"%(iter, n_iters, loss_type, np.mean(losses), np.mean(all_losses)))
            losses = []

    print("iter %d of %d -> Total loss: %.4f, Avg loss: %.4f"%(iter, n_iters, np.mean(losses), np.mean(all_losses)))
    writer.close()

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
def test(model, dg, batch_sz = 8, use_scaled_depth=False, verbose=False):
    model.eval()

    intrinsics = np.array([[768.1605834960938, 0.0, 480.0], [0.0, 768.1605834960938, 270.0], [0.0, 0.0, 1.0]])
    factor_depth = 10000
    data = dg.next_batch(batch_sz)
    image_list, labels_list, mask_list, depth_list, annots = data

    # convert to tensors
    image_tensor, labels_tensor, mask_tensor, depth_tensor = dg.convert_data_batch_to_tensor(data, use_cuda=True)

    preds = model(image_tensor)
    out_channels = preds.size(1)
    class_agnostic = False
    if out_channels == 1:
        class_agnostic = True
    preds = preds.detach().cpu().numpy()#.squeeze()

    for ix,depth_pred in enumerate(preds):
        ann = annots[ix]
        img_id = ann['image_id']
        cls = ann['category_id']

        if not class_agnostic:
            depth_pred = depth_pred[cls]
        else:
            depth_pred = depth_pred[0]

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

        if use_scaled_depth:
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
        dummy_dp = np.zeros(dp.shape, dtype=np.float32) + centroid_z
        dp = paste_mask_on_image(dp, bbox, height, width, interp=cv2.INTER_LINEAR)
        dummy_dp = paste_mask_on_image(dummy_dp, bbox, height, width, interp=cv2.INTER_LINEAR)

        render_predicted_depths(img, intrinsics, depth.astype(np.float32) / factor_depth, pred_depths = [dp, dummy_dp], 
                pred_colors=[get_random_color(), (255,0,0)], factor_depth=1)

class ConvNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(ConvNet, self).__init__()

        conv1_filters = 64
        conv2_filters = 128
        conv3_filters = 256

        self.conv1 = nn.Conv2d(in_channels, conv1_filters, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(conv1_filters, conv2_filters, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(conv2_filters, conv3_filters, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(conv1_filters)
        self.bn2 = nn.BatchNorm2d(conv2_filters)
        self.bn3 = nn.BatchNorm2d(conv3_filters)

        conv_t_filters = 512
        self.conv_t1 = conv_transpose2d_by_factor(conv3_filters, conv_t_filters, factor=2)
        self.conv_t2 = conv_transpose2d_by_factor(conv_t_filters, conv_t_filters, factor=2)
        self.conv_t3 = conv_transpose2d_by_factor(conv_t_filters, conv_t_filters, factor=2)
        self.depth_reg = nn.Conv2d(conv_t_filters, 64, kernel_size=5, stride=1, padding=5 // 2)
        self.depth_reg2 = nn.Conv2d(64, out_channels, kernel_size=5, stride=1, padding=5 // 2)

    def forward(self, x):
        batch_sz = len(x)
        c1 = F.relu(self.bn1(self.conv1(x)))
        c2 = F.relu(self.bn2(self.conv2(c1)))
        c3 = F.relu(self.bn3(self.conv3(c2)))
        # c1 = F.relu(self.conv1(x))
        # c2 = F.relu(self.conv2(c1))
        # c3 = F.relu(self.conv3(c2))
        ct1 = F.relu(self.conv_t1(c3))
        ct2 = F.relu(self.conv_t2(ct1))
        ct3 = F.relu(self.conv_t3(ct2))
        out = F.relu(self.depth_reg(ct3))
        out = self.depth_reg2(out)
        return torch.tanh(out)

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
    CLASS_AGNOSTIC = False

    out_channels = 1 if CLASS_AGNOSTIC else num_classes

    root_dir = "./datasets/FAT/data"
    # ann_file = "./datasets/FAT/coco_fat_debug_1000.json"
    ann_file = "./datasets/FAT/coco_fat_mixed_temple_0.json"

    data_loader = FATDataLoader(root_dir, ann_file, USE_SCALED_DEPTH)
    # data = data_loader.next_batch(2)
    # image_list, labels_list, mask_list, depth_list, annots = data
    # T_im, T_labels, T_seg, T_depth = data_loader.convert_data_batch_to_tensor(data)

    # max_depth = 5
    # for img, mask, depth in zip(image_list, mask_list, depth_list):
    #     cv2.imshow("depth", normalize(depth, 0, max_depth))
    #     cv2.imshow("mask", mask)

    if USE_SCALED_DEPTH:
        save_path = "./model_depth_0.pth"
    else:
        save_path = "./model_noscale_depth_0.pth"
    # save_path = save_path.replace(".pth", "_notanh.pth")
    if CLASS_AGNOSTIC:
        save_path = save_path.replace(".pth", "_agn.pth")

    model = ConvNet(in_channels=3, out_channels=out_channels)
    model.cuda()

    save_path = "model_depth_l1_8k.pth"
    model.load_state_dict(torch.load(save_path))
    print("Loaded %s"%(save_path))

    n_iters = 5000
    lr = 1e-3
    # train(model, data_loader, n_iters=n_iters, lr=lr)
    # torch.save(model.state_dict(), save_path)

    # test_ann_file = "./datasets/FAT/coco_fat_debug_200.json"
    test_ann_file = "./datasets/FAT/coco_fat_mixed_temple_1_n100.json"
    test_data_loader = FATDataLoader(root_dir, test_ann_file, USE_SCALED_DEPTH)    
    # test_data_loader = data_loader
    test(model, test_data_loader, use_scaled_depth=USE_SCALED_DEPTH, batch_sz=16)
