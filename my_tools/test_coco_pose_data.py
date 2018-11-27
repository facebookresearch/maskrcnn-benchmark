import torch
# from torch.distributed import deprecated as dist
import numpy as np
import cv2

from maskrcnn_benchmark.config import cfg

from maskrcnn_benchmark.data.datasets import coco, coco_pose
from maskrcnn_benchmark.data.collate_batch import BatchCollator
from maskrcnn_benchmark.data.build import make_data_sampler, make_batch_data_sampler
import maskrcnn_benchmark.data.transforms as T


shuffle = True
is_distributed = False # gpus > 1
images_per_batch = 2
num_gpus = 1
start_iter = 0
num_iters = 1
aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []
num_workers = 0 #cfg.DATALOADER.NUM_WORKERS  # does not include main thread
images_per_gpu = images_per_batch // num_gpus
is_train = 1

transforms = T.Compose([T.ToTensor()]) # T.build_transforms(cfg, is_train)
remove_images_without_annotations = True

# root = "./datasets/coco/val2014"
# ann_file = "./datasets/coco/annotations/instances_debug2014.json" 
# dataset = coco.COCODataset(ann_file, root, remove_images_without_annotations, transforms)
root = "../PoseCNN/data/LOV/data"
ann_file = "../PoseCNN/coco_ycb_debug.json" 
dataset = coco_pose.COCOPoseDataset(ann_file, root, remove_images_without_annotations, transforms)
# dataset = coco.COCODataset(ann_file, root, remove_images_without_annotations, transforms)

sampler = make_data_sampler(dataset, shuffle, is_distributed)
batch_sampler = make_batch_data_sampler(dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter)
collator = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
data_loader = torch.utils.data.DataLoader(
    dataset,
    num_workers=num_workers,
    batch_sampler=batch_sampler,
    collate_fn=collator,
)

# MODEL
# model = build_detection_model(cfg)
# device = torch.device(cfg.MODEL.DEVICE)
# model.to(device)

def normalize(x, xmin=None, xmax=None):
    xmin = np.min(x) if xmin is None else xmin
    xmax = np.max(x) if xmax is None else xmax
    nx = x - xmin
    if (xmax - xmin) > 0:
        nx /= (xmax - xmin)
    return nx

# def visualize_mask(m):
#     cv2.imshow("mask", m)

def visualize_vertex_centers(vertex_centers):
    # min_depth = np.log(0.3)  # 0.3 m
    # max_depth = np.log(8)  # 8 m
    min_depth = 0
    max_depth = 5
    cx = normalize(vertex_centers[:,:,0],-1,1)
    cy = normalize(vertex_centers[:,:,1],-1,1)
    cz = np.exp(vertex_centers[:,:,2])
    cz[cz==1] = 0
    cz = normalize(cz, min_depth, max_depth)
    cv2.imshow("center x", cx)
    cv2.imshow("center y", cy)
    cv2.imshow("center z", cz)
    # return vertex_centers#, vertex_weights

FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1
FLIP_MODE = FLIP_LEFT_RIGHT

for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
    print(targets)

    id = 0
    im1 = images.tensors[id]
    im_np = im1.numpy()
    im_np = np.transpose(im_np, [1,2,0])[:,:,::-1]  # C,H,W, to H,W,C, then RGB to BGR
    im_copy = im_np.copy()
    h,w,_ = im_np.shape
    cv2.imshow("im", im_np)

    t1 = targets[id]
    # polygons = [x for x in m1]
    # p1 = polygons[0]
    proposal = [w//4,h//4,w//4*3,h//4*3]
    im_copy = cv2.rectangle(im_copy, tuple(proposal[:2]), tuple(proposal[2:]), (0,255,0), 2)
    cv2.imshow("im", im_copy)
    cv2.imshow("flip", cv2.flip(im_copy,1-FLIP_MODE))
    cv2.waitKey(0)

    resize_shape = (w//4, h//4)

    v_field = t1.get_field("vertex")
    m_field = t1.get_field("masks")
    v_field = v_field.crop(proposal)
    m_field = m_field.crop(proposal)
    v_field = v_field.resize(resize_shape)
    m_field = m_field.resize(resize_shape)
    v_field = v_field.transpose(FLIP_MODE)
    m_field = m_field.transpose(FLIP_MODE)

    for ix,vc in enumerate(v_field):
        p = m_field.polygons[ix]
        m = p.convert('mask')
        vc_np = np.transpose(vc.numpy(), [1,2,0])
        visualize_vertex_centers(vc_np)
        # visualize_mask(m.numpy())
        m = m.numpy()  # uint8 format, 0-1
        m *= 255
        cv2.imshow("mask", m)
        cv2.waitKey(0)

