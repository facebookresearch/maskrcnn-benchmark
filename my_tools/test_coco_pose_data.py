import torch
# from torch.distributed import deprecated as dist
import numpy as np
import cv2

from transforms3d.quaternions import quat2mat#, mat2quat
import open3d

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
    # min_depth = 0
    # max_depth = 5
    cx = normalize(vertex_centers[:,:,0],-1,1)
    cy = normalize(vertex_centers[:,:,1],-1,1)
    # cz = np.exp(vertex_centers[:,:,2])
    # cz[cz==1] = 0
    # cz = normalize(cz, min_depth, max_depth)
    cv2.imshow("center x", cx)
    cv2.imshow("center y", cy)
    # cv2.imshow("center z", cz)
    # return vertex_centers#, vertex_weights

def get_random_color():
    return (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))

def get_2d_projected_points(points, intrinsic_matrix, M):
    x3d = np.ones((4, len(points)), dtype=np.float32)
    x3d[0, :] = points[:,0]
    x3d[1, :] = points[:,1]
    x3d[2, :] = points[:,2]

    # projection
    RT = M[:3,:]
    x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
    x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
    x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
    x = np.transpose(x2d[:2,:], [1,0]).astype(np.int32)
    return x


def vis_pose(im, labels, poses, centers, intrinsics, points):

    img = im.copy()

    N = poses.shape[0]

    fx = intrinsics[0,0]
    fy = intrinsics[1,1]
    px = intrinsics[0,2]
    py = intrinsics[1,2]

    colors = [get_random_color() for i in range(N)]
    for i in range(N):
        cls = labels[i]
        center = tuple(centers[i])
        if cls > 0:

            # projection
            RT = np.zeros((3, 4), dtype=np.float32)
            RT[:3, :3] = quat2mat(poses[i, :4])
            dist = poses[i, -1] # z is dist of centroid
            rx = (center[0] - px) / fx
            ry = (center[1] - py) / fy
            RT[:, 3] = np.array([rx * dist, ry * dist, dist])

            proj_pts = get_2d_projected_points(points[cls], intrinsics, RT)
            color = colors[i]

            for pt in proj_pts:
                img = cv2.circle(img, tuple(pt), 1, (int(color[0]),int(color[1]),int(color[2])), -1)

            # plt.plot(x2d[0, :], x2d[1, :], '.', , alpha=0.5)
            # plt.scatter(x2d[0, :], x2d[1, :], marker='o', color=np.divide(colors[cls], 255.0), s=10)
    cv2.imshow("proj points", img)

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

    cfg.INPUT.MIN_SIZE_TRAIN = 240 * 2
    cfg.INPUT.MAX_SIZE_TRAIN = 320 * 2
    cfg.INPUT.FLIP_PROB_TRAIN = 0
    cfg.MODEL.VERTEX_ON = True
    cfg.MODEL.POSE_ON = True
    cfg.MODEL.DEPTH_ON = True

    # transforms = T.Compose([T.ToTensor()]) # T.build_transforms(cfg, is_train)
    transforms = T.build_transforms(cfg, is_train, normalize=False)

    # root = "/home/bot/hd/datasets/MSCOCO/val2014"
    # ann_file = "/home/bot/hd/datasets/MSCOCO/annotations/instances_debug2014.json"
    # dataset = coco.COCODataset(ann_file, root, remove_images_without_annotations, transforms)
    # root = "./datasets/LOV/data"
    # ann_file = "./datasets/LOV/coco_lov_debug.json"
    root = "./datasets/FAT/data"
    ann_file = "./datasets/FAT/coco_fat_debug.json"
    # ann_file = "./datasets/FAT/coco_fat_mixed_temple_0.json"

    # root = "/home/bot/LabelMe/Images/loading"
    # ann_file = "/home/bot/LabelMe/scripts/loading.json"
    dataset = coco_pose.COCOPoseDataset(ann_file, root, remove_images_without_annotations, transforms, cfg)

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

    if cfg.MODEL.POSE_ON:
        # intrinsics = np.array([[1066.778, 0, 312.9869], [0, 1067.487, 241.3109], [0.0, 0.0, 1.0]])
        intrinsics = np.array([[768.16,0,480],[0,768.16,270],[0,0,1]])

        points_file = "./datasets/FAT/points_all_orig.npy"  
        points = np.load(points_file)
        # points = [[]]
        # for cls in CLASSES[1:]:
        #     points.append(np.loadtxt(root + "/../models/%s/points.xyz"%(cls)))
        # np.save(points_file, points)

        # coord_frame = open3d.create_mesh_coordinate_frame(size = 0.6, origin = [0, 0, 0])
        # for pts in points[1:]:
        #     open3d.draw_geometries([create_cloud(pts), coord_frame])

    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        print(targets)

        for id in range(len(targets)):
            t1 = targets[id]
            im_scale = t1.get_field("scale")

            if cfg.MODEL.POSE_ON or cfg.MODEL.DEPTH_ON:
                try:
                    intrinsics = t1.get_field("intrinsic_matrix")
                except KeyError:
                    print("[WARN]: Could not get 'intrinsic_matrix' field, using default...")

            im1 = images.tensors[id]
            im_np = im1.numpy()
            im_np = np.transpose(im_np, [1,2,0])[:,:,::-1]  # C,H,W, to H,W,C, then RGB to BGR
            im_copy = (im_np * 255).astype(np.uint8)
            h,w,_ = im_np.shape
            cv2.imshow("im", im_np)

            m_field = t1.get_field("masks")
            labels = t1.get_field('labels')
            
            if cfg.MODEL.VERTEX_ON or cfg.MODEL.POSE_ON:
                c_field = t1.get_field("centers")
                centers = np.array([c.keypoints[0].numpy()[:,:2] for c in c_field]).squeeze()

                if cfg.MODEL.VERTEX_ON:
                    v_field = t1.get_field("vertex")

                if cfg.MODEL.POSE_ON:
                    poses = t1.get_field("poses")
                    K = intrinsics * im_scale
                    K[-1,-1] = 1
                    poses = poses.numpy()
                    vis_pose(im_copy, labels, poses, centers, K, points)

            if cfg.MODEL.DEPTH_ON:
                depth_field = t1.get_field("depth")

            for ix,label in enumerate(labels):
                print("Label: %d, %s"%(label, CLASSES[label]))
                p = m_field.instances.polygons[ix]
                m = p.convert_to_binarymask()
                # visualize_mask(m.numpy())
                m = m.numpy()  # uint8 format, 0-1
                m *= 255
                m = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
                
                if cfg.MODEL.VERTEX_ON or cfg.MODEL.POSE_ON:
                    center = centers[ix]
                    m = cv2.circle(m, (int(center[0]), int(center[1])), 2, (0,255,0), -1)

                    if cfg.MODEL.VERTEX_ON:
                        vc_np = np.transpose(v_field[ix].data.numpy().squeeze(), [1,2,0])
                        visualize_vertex_centers(vc_np)

                if cfg.MODEL.DEPTH_ON:
                    depth_np = depth_field[ix].data.numpy().squeeze()
                    max_depth = 7 
                    depth_np = 1.0 - normalize(depth_np, 0, max_depth)
                    cv2.imshow("depth", depth_np)

                cv2.imshow("mask", m)
                cv2.waitKey(0)

