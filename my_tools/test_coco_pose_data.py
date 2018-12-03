import torch
# from torch.distributed import deprecated as dist
import numpy as np
import cv2

from maskrcnn_benchmark.config import cfg

from maskrcnn_benchmark.data.datasets import coco, coco_pose
from maskrcnn_benchmark.data.collate_batch import BatchCollator
from maskrcnn_benchmark.data.build import make_data_sampler, make_batch_data_sampler
import maskrcnn_benchmark.data.transforms as T

def normalize(x, xmin=None, xmax=None):
    xmin = np.min(x) if xmin is None else xmin
    xmax = np.max(x) if xmax is None else xmax
    nx = x - xmin
    if (xmax - xmin) > 0:
        nx /= (xmax - xmin)
    return nx

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

CATEGORIES = [
    '__background__', '002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can',
     '006_mustard_bottle', \
     '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana',
     '019_pitcher_base', \
     '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors',
     '040_large_marker', \
     '051_large_clamp', '052_extra_large_clamp', '061_foam_brick'
]

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
remove_images_without_annotations = True

cfg.INPUT.MIN_SIZE_TRAIN = 240
cfg.INPUT.MAX_SIZE_TRAIN = 320

# transforms = T.Compose([T.ToTensor()]) # T.build_transforms(cfg, is_train)
transforms = T.build_transforms(cfg, is_train)

# root = "./datasets/coco/val2014"
# ann_file = "./datasets/coco/annotations/instances_debug2014.json" 
# dataset = coco.COCODataset(ann_file, root, remove_images_without_annotations, transforms)
root = "./datasets/LOV/data"
ann_file = "./datasets/LOV/coco_lov_debug.json"
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

FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1
FLIP_MODE = FLIP_LEFT_RIGHT

for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
    print(targets)

    for t1 in targets:
        print(t1.get_field("scale"))

    for id in range(len(targets)):
        t1 = targets[id]

        im1 = images.tensors[id]
        im_np = im1.numpy()
        im_np = np.transpose(im_np, [1,2,0])[:,:,::-1]  # C,H,W, to H,W,C, then RGB to BGR
        im_copy = im_np.copy()
        h,w,_ = im_np.shape
        cv2.imshow("im", im_np)

        v_field = t1.get_field("vertex")
        m_field = t1.get_field("masks")
        labels = t1.get_field('labels')

        # proposal = [w//4,h//4,w//4*3,h//4*3]
        # im_copy = cv2.rectangle(im_copy, tuple(proposal[:2]), tuple(proposal[2:]), (0,255,0), 2)
        # cv2.imshow("im", im_copy)
        # cv2.imshow("flip", cv2.flip(im_copy,1-FLIP_MODE))
        # cv2.waitKey(0)
        # v_field = v_field.crop(proposal)
        # m_field = m_field.crop(proposal)
        # resize_shape = (w//4, h//4)
        # v_field = v_field.resize(resize_shape)
        # m_field = m_field.resize(resize_shape)
        # v_field = v_field.transpose(FLIP_MODE)
        # m_field = m_field.transpose(FLIP_MODE)

        for ix,vc in enumerate(v_field):
            label = labels[ix]
            print("Label: %d, %s"%(label, CATEGORIES[label]))
            p = m_field.polygons[ix]
            m = p.convert('mask')
            # vc_np = np.transpose(vc.numpy(), [1,2,0])
            # visualize_vertex_centers(vc_np)
            # visualize_mask(m.numpy())
            m = m.numpy()  # uint8 format, 0-1
            m *= 255
            cv2.imshow("mask", m)
            cv2.waitKey(0)

