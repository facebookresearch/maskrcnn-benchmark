import numpy as np
import cv2
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import open3d

from transforms3d.quaternions import quat2mat#, mat2quat


def create_cloud(points, normals=[], colors=[], T=None):
    cloud = open3d.PointCloud()
    cloud.points = open3d.Vector3dVector(points)
    if len(normals) > 0:
        assert len(normals) == len(points)
        cloud.normals = open3d.Vector3dVector(normals)
    if len(colors) > 0:
        # if len(colors) == 1: # ((1,3))
        #     colors = np.array(colors) * len(points)
        # else:
        assert len(colors) == len(points)
        cloud.colors = open3d.Vector3dVector(colors)

    if T is not None:
        cloud.transform(T)
    return cloud

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


def average_point_distance_metric(points, rotation1, rotation2, closest_point=False):
    assert rotation1.shape == rotation2.shape == (3,3)
    if closest_point:
        M1 = np.identity(4)
        M1[:3,:3] = rotation1
        M2 = np.identity(4)
        M2[:3,:3] = rotation2
        cloud1 = create_cloud(points, T=M1)
        cloud2 = create_cloud(points, T=M2)

        # get closest point
        tree = open3d.KDTreeFlann(cloud1)
        distances = []
        for ix in range(len(points)):
            _, idx, _ = tree.search_knn_vector_3d(cloud2.points[ix], 1)
            idx = idx.pop()#; dist = dist.pop()
            dist = np.linalg.norm(cloud2.points[ix] - cloud1.points[idx])
            distances.append(dist)
        return np.mean(distances)
        # # use icp to compare distance of nearest points
        # reg = open3d.registration_icp(cloud1, cloud2, 0.02, np.identity(4),
        #     open3d.TransformationEstimationPointToPoint())
        # inliers = np.asarray(reg.correspondence_set)
        # p1 = np.asarray(cloud1.points)
        # p2 = np.asarray(cloud2.points)
    else:
        pts = np.array(points).T if isinstance(points, list) else points.T
        p1 = np.dot(rotation1, pts).T
        p2 = np.dot(rotation2, pts).T
        return np.linalg.norm(p1 - p2, axis=1).mean()

class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None

class Conv2d(torch.nn.Conv2d):
    def forward(self, x):
        if x.numel() > 0:
            return super(Conv2d, self).forward(x)
        # get output shape

        output_shape = [
            (i + 2 * p - (di * (k - 1) + 1)) // d + 1
            for i, p, di, k, d in zip(
                x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
            )
        ]
        output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)

class ConvTranspose2d(torch.nn.ConvTranspose2d):
    def forward(self, x):
        if x.numel() > 0:
            return super(ConvTranspose2d, self).forward(x)
        # get output shape

        output_shape = [
            (i - 1) * d - 2 * p + (di * (k - 1) + 1) + op
            for i, p, di, k, d, op in zip(
                x.shape[-2:],
                self.padding,
                self.dilation,
                self.kernel_size,
                self.stride,
                self.output_padding,
            )
        ]
        output_shape = [x.shape[0], self.bias.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)

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

def conv_transpose2d_by_factor(in_cn, out_cn, factor):
    """
    Maintain output_size = input_size * factor (multiple of 2)
    """
    # stride = int(1.0/spatial_scale)
    assert factor >= 2 and factor % 2 == 0
    stride = factor
    k = stride * 2
    kernel_size = (k,k)
    p = stride // 2
    padding = (p, p)
    stride = (stride, stride)
    return ConvTranspose2d(in_cn, out_cn, kernel_size, stride, padding)


class DataGenerator():
    def __init__(self):
        IMG_SIZE = 56
        self.H = IMG_SIZE
        self.W = IMG_SIZE

    def next_batch(self, batch_size=8):
        data = [self._get_random_data() for i in range(batch_size)]
        return data

    def _get_random_data(self, depth=None):

        sz = (self.H, self.W)
        m = np.zeros((self.H, self.W, 3), dtype=np.float32)
        m_gt = np.zeros(sz, dtype=np.float32)

        if depth is None:
            depth = float(np.random.randint(-50,50)) / 10

        m_gt[:] = depth 

        # set first channel to depth values + gaussian noise
        mean = 0
        std = 0.03
        m[:,:,0] = depth + np.random.normal(mean, std, size=sz)
        # set second and third channels as random noise
        m[:,:,1] = np.random.random(size=sz)
        m[:,:,2] = np.random.random(size=sz)

        return [m, m_gt, depth]

    def convert_data_batch_to_tensor(self, data, use_cuda=False):
        m_data = []
        m_gt_data = []

        for d in data:
            m = d[0]
            m = np.transpose(m, [2,0,1])
            m_data.append(m)
            m_gt_data.append(d[1])

        tm = torch.FloatTensor(m_data)#.unsqueeze(0)
        tmgt = torch.FloatTensor(m_gt_data).unsqueeze(1)

        if use_cuda:
            tm = tm.cuda()
            tmgt = tmgt.cuda()
        return tm, tmgt

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

class ConvNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(ConvNet, self).__init__()

        conv1_filters = 64
        conv2_filters = 128
        conv3_filters = 256
        conv4_filters = 512
        conv5_filters = 1024

        self.conv1 = nn.Conv2d(in_channels, conv1_filters, kernel_size=5, stride=2, padding=5//2)
        self.conv2 = nn.Conv2d(conv1_filters, conv2_filters, kernel_size=5, stride=1, padding=5//2)
        self.conv3 = nn.Conv2d(conv2_filters, conv3_filters, kernel_size=5, stride=2, padding=5//2)
        self.conv4 = nn.Conv2d(conv3_filters, conv4_filters, kernel_size=3, stride=1, padding=3//2)
        self.conv5 = nn.Conv2d(conv4_filters, conv5_filters, kernel_size=3, stride=1, padding=3 // 2)
        self.bn1 = nn.BatchNorm2d(conv1_filters)
        self.bn2 = nn.BatchNorm2d(conv2_filters)
        self.bn3 = nn.BatchNorm2d(conv3_filters)
        self.bn4 = nn.BatchNorm2d(conv4_filters)
        self.bn5 = nn.BatchNorm2d(conv5_filters)

        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)

        self.reg = nn.Linear(conv5_filters, out_channels)
        # self.reg = nn.Conv2d(conv2_filters, out_channels, kernel_size=3, stride=1, padding=3 // 2)
        self._init_params()

    def _init_params(self):
        conv_modules = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5] #self.conv_t1, self.conv_t2, self.conv_t3]
        for m in conv_modules:
            nn.init.constant_(m.bias, 0)
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            # nn.init.normal_(m.weight, mean=0, std=0.001)

        fc_modules = [self.reg]
        for m in fc_modules:
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # batch_sz = len(x)
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = self.max_pool(c1)
        c2 = F.relu(self.bn2(self.conv2(c1)))
        c3 = F.relu(self.bn3(self.conv3(c2)))
        c4 = F.relu(self.bn4(self.conv4(c3)))
        c5 = F.relu(self.bn5(self.conv5(c4)))

        x = self.avgpool(c5)
        x = x.view(x.size(0), -1)
        out = self.reg(x)
        return out # torch.tanh(out)

def l1_loss(x, y):
    return torch.abs(x - y)

def mean_var_loss(input, target):
    
    diff = target - input
    md = torch.mean(diff)
    var_loss = (1 + torch.abs(diff - md)) ** 2 - 1
    # var_loss = (input - torch.mean(input))

    loss = 0.5 * torch.abs(diff) + 0.5 * var_loss

    # mean_p = torch.mean(input)
    # mean_t = torch.mean(target)
    # mean_loss = torch.abs(mean_t - mean_p)
    # loss = 0.5 * mean_loss + 1.5 * var_loss

    return loss

# @persistent_locals
def train(model, dg):

    epochs = 10
    n_iters = 1000
    batch_size = 32
    lr = 1e-3

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    losses = []
    for iter in range(n_iters):
        data = dg.next_batch(batch_size)

        train_x, train_y = dg.convert_data_batch_to_tensor(data, use_cuda=True)
        optimizer.zero_grad()

        output = model(train_x)

        # loss
        # loss_type = "l1"
        # loss = l1_loss(output, train_y).mean()
        loss_type = "mean_var"
        loss = mean_var_loss(output, train_y).mean()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if iter % 20 == 0 and iter > 0:
            print("iter %d of %d -> Total loss (%s): %.3f"%(iter, n_iters, loss_type, np.mean(losses)))
            losses = []

    print("iter %d of %d -> Total loss: %.3f"%(iter, n_iters, loss.item()))

# @persistent_locals
def test(model, dg, batch_sz = 8, verbose=False):
    model.eval()
    # batch_sz = 100
    # test_data = dg.next_batch(test_batch_sz)
    test_data = [dg._get_random_data(float(np.random.randint(-10,10)) / 14) for i in range(batch_sz)]
    test_x, test_y = dg.convert_data_batch_to_tensor(test_data, use_cuda=True)

    preds = model(test_x)
    preds = preds.detach().cpu().numpy()

    mean_arr = []
    std_arr = []
    for ix,p in enumerate(preds):
        gt = test_data[ix][-1]
        err = np.mean(np.abs(p - gt))
        # x = test_data[ix][0][:,:,0]
        # x_mean = np.mean(x)
        # x_std = np.std(x)
        p_mean = np.mean(p)
        p_std = np.std(p)
        err_diff = gt - p_mean
        mean_arr.append(np.abs(err_diff))
        std_arr.append(p_std)
        if verbose:
            print("Abs error: %.3f, GT: %.3f, Pred Mean: %.3f (err diff: %.3f), Pred Std: %.3f"%(err, gt, p_mean, err_diff, p_std))
    print("Batch size: %d -> Average mean err: %.3f, Average std: %.3f"%(batch_sz, np.mean(mean_arr), np.mean(std_arr)))

if __name__ == '__main__':
    dg = DataGenerator()
    batch_sz = 8
    data = dg.next_batch(batch_sz)
    x = dg.convert_data_batch_to_tensor(data)

    model = ConvNet(in_channels=3, out_channels=1)
    model.cuda()
    print("Model constructed")

    train(model, dg)
    test(model, dg, 200, False)
    # test(model, dg, 10, verbose=True)
