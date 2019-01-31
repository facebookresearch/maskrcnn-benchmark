import cv2
import numpy as np
import numpy.random as npr
import os
import os.path as osp
import glob
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from resnet import ResNet50
from vgg16 import VGG16

PIXEL_MEAN = [102.9801, 115.9465, 122.7717]

def FT(x): return torch.FloatTensor(x)
def LT(x): return torch.LongTensor(x)

def _get_conv_transpose2d_by_factor(in_cn, out_cn, factor):
    """
    Maintain output_size = input_size * factor (multiple of 2)
    """
    # stride = int(1.0/spatial_scale)
    assert factor >= 2 and factor % 2 == 0
    stride = factor
    k = stride * 2
    kernel_size = (k,k)
    padding = stride // 2
    stride = (stride, stride)
    return _get_conv_transpose2d(in_cn, out_cn, kernel_size, stride, padding)#, trainable=trainable)

def _get_conv_transpose2d(in_cn, out_cn, kernel_size, stride=(1,1), padding=0, trainable=False):
    x = nn.ConvTranspose2d(in_cn, out_cn, kernel_size=kernel_size, stride=stride, padding=padding)
    x.bias.data.zero_()
    # if not trainable:
    #     for p in x.parameters():
    #         p.requires_grad = False
    return x

def create_log_label_weights(log_label):
    assert len(log_label.shape) == 2
    h,w = log_label.shape
    instance_weight = np.zeros((h,w), np.float32)
    sumP = len(log_label[log_label==1])
    sumN = h*w - sumP
    # 'balanced' case only for instance weights
    weightP = 0.5 / sumP if sumP != 0 else 0
    weightN = 0.5 / sumN if sumN != 0 else 0
    instance_weight[log_label==1] = weightP
    instance_weight[log_label!=1] = weightN
    return instance_weight


class DataLoader(object):
    def __init__(self, img_dir, labels_dir, shuffle=True):
        self.img_dir = img_dir 
        self.labels_dir = labels_dir # 

        self.masks = glob.glob(self.labels_dir + '/*.png')
        self.total_cnt = len(self.masks)

        self.shuffle = shuffle
        self.cur_idx = 0
        self.permutation = None
        self._reset_permutation()

    def _reset_permutation(self):
        total = self.total_cnt
        self.permutation = npr.permutation(total) if self.shuffle else np.arange(total)

    def _get_image_path(self, mask_path):
        base_path = osp.basename(mask_path)
        img_path = osp.join(self.img_dir, base_path.replace(".png", ".jpg"))
        return img_path

    def _get_mask_path(self, image_path):
        base_path = osp.basename(image_path)
        mask_path = osp.join(self.labels_dir, base_path.replace(".jpg", ".png"))
        return mask_path

    def get_next_batch_perm(self, batch_sz):
        start_idx = self.cur_idx
        self.cur_idx += batch_sz
        perm = self.permutation[start_idx:self.cur_idx]
        if self.cur_idx >= self.total_cnt:
            self._reset_permutation()
            self.cur_idx -= self.total_cnt
            perm = np.hstack((perm, self.permutation[:self.cur_idx]))
        return perm

    def next_batch(self, batch_sz):
        perm = self.get_next_batch_perm(batch_sz)
        masks = [self.masks[idx] for idx in perm]

        image_list = []
        mask_list = []
        for m_path in masks:
            m = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
            if m is None:
                print("Could not find %s"%(m_path))
                continue

            m = m.astype(np.float32) / 255  # assumes mask is only 0 or 255!

            im_path = self._get_image_path(m_path)
            im = cv2.imread(im_path)
            if im is None:
                print("Could not find %s"%(im_path))
                continue

            image_list.append(im)
            mask_list.append(m)

        return image_list, mask_list

    def convert_data_batch_to_tensor(self, data, resize_height=480, resize_width=800, use_cuda=False):
        image_list, mask_list = data
        N = len(image_list)

        if N == 0:
            return []

        t_image_list = np.zeros((N, resize_height, resize_width, 3), dtype=np.float32)
        t_mask_list = np.zeros((N, resize_height, resize_width), dtype=np.float32)
        t_mask_weights_list = t_mask_list.copy()
        for ix, im in enumerate(image_list):
            
            # h,w = im.shape[:2]
            t_im = cv2.resize(im, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
            t_im = t_im.astype(np.float32) / 255  # assumes 0-255!
            t_image_list[ix] = t_im

            t_mask = cv2.resize(mask_list[ix], (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
            t_mask[t_mask >= 0.5] = 1
            t_mask[t_mask < 0.5] = 0
            t_mask_list[ix] = t_mask

            t_mask_weights_list[ix] = create_log_label_weights(t_mask)

        t_image_list = np.transpose(t_image_list, [0,3,1,2])  # (N,H,W,3) to (N,3,H,W)
        t_image_tensor = FT(t_image_list)
        t_mask_tensor = FT(t_mask_list)
        t_mask_weights_tensor = FT(t_mask_weights_list)
        if use_cuda:
            t_image_tensor = t_image_tensor.cuda()
            t_mask_tensor = t_mask_tensor.cuda()
            t_mask_weights_tensor = t_mask_weights_tensor.cuda()

        return t_image_tensor, t_mask_tensor, t_mask_weights_tensor

    def visualize(self, data):
        image_list, mask_list = data
        for ix,im in enumerate(image_list):
            cv2.imshow("im", im)
            cv2.imshow("mask", mask_list[ix])
            cv2.waitKey(0)

class SemSeg(nn.Module):

    def __init__(self, out_channels=1):
        super(SemSeg, self).__init__()

        self.features = VGG16()  # ResNet50() 
        dim_outs = self.features.conv_dim_out
        dim_out_0, dim_out_1 = dim_outs[-2:]
        spatial_scales = self.features.conv_spatial_scale
        spatial_scale_0, spatial_scale_1 = spatial_scales[-2:]

        num_units = 64
        inplace = True
        self.score_conv4 = nn.Sequential(
            nn.Conv2d(dim_out_0, num_units, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace)
        )   
        self.score_conv5 = nn.Sequential(
            nn.Conv2d(dim_out_1, num_units, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace)
        )
        self.upscore_conv5 = _get_conv_transpose2d_by_factor(num_units, num_units, factor=int(spatial_scale_0/spatial_scale_1))
        self.upscore = _get_conv_transpose2d_by_factor(num_units, num_units, factor=int(1.0/spatial_scale_0))
        self.score = nn.Sequential(
            nn.Conv2d(num_units, out_channels, kernel_size=(1, 1), stride=(1, 1)),
            # nn.ReLU(inplace)
        )

    def forward(self, x):
        c4, c5 = self.features(x)
        s4 = self.score_conv4(c4)
        s5 = self.score_conv5(c5)
        us5 = self.upscore_conv5(s5)
        add = torch.add(s4, us5)
        upsc = self.upscore(add)
        sc = self.score(upsc)
        # softmax = F.softmax(sc, 1)
        return sc #, softmax

    def load_pretrained(self, model_file, verbose=False):
        self.features.load_pretrained(model_file, verbose)

def train(model, dg, n_iters=1000, batch_sz = 4, lr=1e-3):
    # from tensorboardX import SummaryWriter
    # writer = SummaryWriter()

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    criterion = nn.BCELoss(reduce=False)

    all_losses = []
    losses = []
    for iter in range(n_iters):
        data = dg.next_batch(batch_sz)

        # normalize the pixels by pixel mean
        image_list = data[0]
        for ix,im in enumerate(image_list):
            image_list[ix] = im.astype(np.float32) - PIXEL_MEAN

        image_tensor, mask_tensor, mask_weights_tensor = dg.convert_data_batch_to_tensor(data, resize_height=480, resize_width=800, use_cuda=True)
        optimizer.zero_grad()

        pred = model(image_tensor)

        if pred.numel() == 0:
            continue

        pred = torch.sigmoid(pred[:,0]) # (N,H,W)

        # loss
        entropy_loss = criterion(pred, mask_tensor)
        weighted_loss = torch.mul(entropy_loss, mask_weights_tensor)
        loss = weighted_loss.sum(dim=(1,2)).mean()
        loss.backward()
        optimizer.step()
        loss_value = loss.item()

        losses.append(loss_value)
        all_losses.append(loss_value)

        # writer.add_scalar('data/seg_binary_entropy_loss', loss_value, iter)

        if iter % 20 == 0 and iter > 0:
            print("iter %d of %d -> Total loss: %.4f, Avg loss: %.4f"%(iter, n_iters, np.mean(losses), np.mean(all_losses)))
            losses = []
            # p = pred[0].detach().cpu().numpy().squeeze()
            # p[p>=0.5]=1
            # p[p<0.5]=0
            # cv2.imshow("train_pred_sample", p)
            # cv2.waitKey(150)

    print("iter %d of %d -> Total loss: %.4f, Avg loss: %.4f"%(iter, n_iters, np.mean(losses), np.mean(all_losses)))
    # writer.close()

def normalize(x, xmin=None, xmax=None):
    xmin = np.min(x) if xmin is None else xmin
    xmax = np.max(x) if xmax is None else xmax
    nx = x - xmin
    nx /= (xmax - xmin)
    return nx

def inference(model, img_dir, use_cuda=False):
    model.eval()

    for img_path in glob.glob(img_dir + "/*.jpg"):
        img = cv2.imread(img_path)
        if img is None:
            print("Could not find %s"%(img_path))
            continue

        h, w = img.shape[:2]

        # normalize the pixels by pixel mean
        im = img.astype(np.float32) - PIXEL_MEAN
        im /= 255
        resize_height = 480
        resize_width = 800

        resize_im = cv2.resize(im, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
        t_im = np.transpose(resize_im, [2,0,1])   # H,W,3 to 3,H,W
        image_tensor = FT([t_im])  # 1,3,H,W
        if use_cuda:
            image_tensor = image_tensor.cuda()

        pred = model(image_tensor)
        if pred.numel() == 0:
            print("MODEL RETURNED EMPTY TENSOR")
            return

        prob = torch.sigmoid(pred)
        # topk = torch.topk(prob)
        prob = prob.detach().cpu().numpy().squeeze()

        # resize back to original image size
        prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = prob.copy()
        mask[mask > 0.5] = 1
        mask[mask < 0.5] = 0
        prob_norm = prob.copy()
        # prob_norm[prob >= 0.8] = normalize(prob[prob >= 0.8], 0.5, 1)
        # prob_norm[prob < 0.8] = normalize(prob[prob < 0.8], 0, 0.5)
        prob_norm_255 = (prob_norm * 255).astype(np.uint8)
        prob_img = cv2.applyColorMap(prob_norm_255, cv2.COLORMAP_JET)
        heatmap = cv2.addWeighted(img, 0.5, prob_img, 0.5, 0)

        p_sorted_idx = np.argsort(prob.flatten())[::-1]

        k = 30
        min_dist_separation = 40  # distance in pixels
        top_idx = p_sorted_idx[0]
        top_sample_pts = [np.array((top_idx % w, top_idx // w))]
        for idx in p_sorted_idx[1:]:
            pt = np.array((idx % w, idx // w))
            valid = True
            for pt2 in top_sample_pts:
                dist = np.linalg.norm(pt2 - pt) # check dist
                if dist < min_dist_separation:
                    valid = False
                    break
            if valid:
                top_sample_pts.append(pt)
                if len(top_sample_pts) == k:
                    break

        for ix, tpt in enumerate(top_sample_pts):
            pt = tuple(tpt)
            color = (0,255,0)
            txt_color = (0,0,0)
            if ix < 5:
                txt_color = (0,0,255)
                color = (0,0,255)
            cv2.putText(heatmap, "%d"%(ix + 1), pt, cv2.FONT_HERSHEY_COMPLEX, 0.5, txt_color)
            cv2.circle(heatmap, pt, 3, color, -1)

        cv2.imshow("im", img)
        cv2.imshow("mask", mask)
        cv2.imshow("pred_probabilities", heatmap)
        cv2.waitKey(0)


if __name__ == '__main__':
    img_dir = "/home/bot/Downloads/Rebin/25_jan_data"  # _debug"
    # img_dir = "/home/bot/Downloads/Rebin/debug"
    labels_dir = osp.join(img_dir, 'labels')
    data_loader = DataLoader(img_dir, labels_dir)
    # data = data_loader.next_batch(8)
    # data_loader.convert_data_batch_to_tensor(data, resize_height=480, resize_width=800)
    # data_loader.visualize(data)

    save_path = "tmp_seg.pth"

    # pretrained_file = "/data/models/resnet50.pth"
    pretrained_file = "/data/models/vgg16.pth"
    model = SemSeg(out_channels=1)
    model.load_pretrained(pretrained_file)
    model.cuda()

    model.load_state_dict(torch.load(save_path))
    print("Loaded %s"%(save_path))

    n_iters = 1000
    lr = 1e-3
    batch_sz = 2
    # train(model, data_loader, n_iters=n_iters, batch_sz = batch_sz, lr=lr)
    # torch.save(model.state_dict(), save_path)

    test_img_dir = "/home/bot/Downloads/Rebin/25_jan_data_test"
    inference(model, test_img_dir, use_cuda=True)

