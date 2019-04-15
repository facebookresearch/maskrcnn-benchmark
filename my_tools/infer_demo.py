from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.data.build import build_transforms
from torchvision.transforms import functional as TVF

import numpy as np
from transforms3d.quaternions import quat2mat#, mat2quat

import torch
import torch.nn.functional as F

import glob
import cv2
import open3d

def load_model(model, f):
    checkpoint = torch.load(f, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint.pop("model"))
    print("Loaded %s"%(f))


class PseudoTarget(object):
    def __init__(self):
        self.x = {}

    def add_field(self, k, v):
        self.x[k] = v

    def get_field(self, k):
        return self.x[k]

    def resize(self, *args):
        return self

    def transpose(self, *args):
        return self

class ImageTransformer(object):
    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.transforms = self.build_transform()

    def build_transform(self):
        transforms = build_transforms(self.cfg, is_train=False)
        return transforms

    def transform_image(self, original_image):
        target = PseudoTarget()
        image = TVF.to_pil_image(original_image, mode=None)
        image, target = self.transforms(image, target)
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image = image.unsqueeze(0)
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        return image_list, [target.get_field('scale')]

def select_top_predictions(predictions, confidence_threshold=0.7):
    """
    Select only predictions which have a `score` > self.confidence_threshold,
    and returns the predictions in descending order of score

    Arguments:
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores`.

    Returns:
        prediction (BoxList): the detected objects. Additional information
            of the detection properties can be found in the fields of
            the BoxList via `prediction.fields()`
    """
    scores = predictions.get_field("scores")
    keep = torch.nonzero(scores > confidence_threshold).squeeze(1)
    predictions = predictions[keep]
    scores = predictions.get_field("scores")
    _, idx = scores.sort(0, descending=True)
    return predictions[idx]

def paste_mask_on_image(mask, box, im_h, im_w, thresh=None, interp=cv2.INTER_LINEAR):
    w = box[2] - box[0] + 1
    h = box[3] - box[1] + 1
    w = max(w, 1)
    h = max(h, 1)

    resized = cv2.resize(mask, (w,h), interpolation=interp)

    if thresh is not None:#thresh >= 0:
        resized = resized > thresh

    x_0 = max(box[0], 0)
    x_1 = min(box[2] + 1, im_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3] + 1, im_h)

    canvas = np.zeros((im_h, im_w), dtype=np.float32)
    canvas[y_0:y_1, x_0:x_1] = resized[(y_0 - box[1]) : (y_1 - box[1]), (x_0 - box[0]) : (x_1 - box[0])]
    return canvas

def normalize(x, xmin=None, xmax=None):
    xmin = np.min(x) if xmin is None else xmin
    xmax = np.max(x) if xmax is None else xmax
    nx = x - xmin
    nx /= (xmax - xmin)
    return nx

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

def vis_pose(im, labels, poses, intrinsics, points):

    img = im.copy()

    N = poses.shape[0]

    colors = [get_random_color() for i in range(N)]
    for i in range(N):
        cls = labels[i]
        if cls > 0:
            # projection
            RT = np.zeros((3, 4), dtype=np.float32)
            RT[:3, :3] = quat2mat(poses[i, :4])
            RT[:, 3] = poses[i, 4:7]

            proj_pts = get_2d_projected_points(points[cls], intrinsics, RT)

            color = colors[i]
            for px in proj_pts:
                img = cv2.circle(img, tuple(px), 1, (int(color[0]),int(color[1]),int(color[2])), -1)

            # plt.plot(x2d[0, :], x2d[1, :], '.', , alpha=0.5)
            # plt.scatter(x2d[0, :], x2d[1, :], marker='o', color=np.divide(colors[cls], 255.0), s=10)
    cv2.imshow("proj points", img)


def vis_rois(im, rois):
    img = im.copy()

    rois = np.round(rois).astype(np.int32)
    RED = (0,0,255)
    GREEN = (0,255,0)
    for roi in rois:
        cx = (roi[2] + roi[4]) // 2
        cy = (roi[3] + roi[5]) // 2
        bbox1 = roi[2:-1]
        img = cv2.rectangle(img, tuple(bbox1[:2]), tuple(bbox1[2:]), RED)
        img = cv2.circle(img, (cx,cy), 3, GREEN, -1)

    cv2.imshow("rois", img)

def backproject_camera(im_depth, meta_data):

    depth = im_depth.astype(np.float32, copy=True) / meta_data['factor_depth']

    # get intrinsic matrix
    K = meta_data['intrinsic_matrix']
    K = np.matrix(K)
    K = np.reshape(K,(3,3))
    Kinv = np.linalg.inv(K)
    # if cfg.FLIP_X:
    #     Kinv[0, 0] = -1 * Kinv[0, 0]
    #     Kinv[0, 2] = -1 * Kinv[0, 2]

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

def render_predicted_depths(im, depth, pred_depths, meta_data):
    rgb = im.copy()
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32)[:,:,::-1] / 255

    X = backproject_camera(depth, meta_data)
    cloud_rgb = rgb # .astype(np.float32)[:,:,::-1] / 255
    cloud_rgb = cloud_rgb.reshape((cloud_rgb.shape[0]*cloud_rgb.shape[1],3))
    scene_cloud = create_cloud(X.T, colors=cloud_rgb)

    md = {'intrinsic_matrix': meta_data['intrinsic_matrix'], 'factor_depth': 1}
    for dp in pred_depths:
        X = backproject_camera(dp, md)
        pred_cloud = create_cloud(X.T)
        open3d.draw_geometries([scene_cloud, pred_cloud])

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

def render_object_pose(im, depth, labels, meta_data, pose_data, points):
    """
    im: rgb image of the scene
    depth: depth image of the scene
    meta_data: dict({'intrinsic_matrix': K, 'factor_depth': })
    pose_data: [{"name": "004_sugar_box", "pose": 3x4 or 4x4 matrix}, {...}, ]
    """
    if len(pose_data) == 0:
        return 

    rgb = im.copy()
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32)[:,:,::-1] / 255

    X = backproject_camera(depth, meta_data)
    cloud_rgb = rgb # .astype(np.float32)[:,:,::-1] / 255
    cloud_rgb = cloud_rgb.reshape((cloud_rgb.shape[0]*cloud_rgb.shape[1],3))
    scene_cloud = create_cloud(X.T, colors=cloud_rgb)

    if len(pose_data) == 0:
        open3d.draw_geometries([scene_cloud])
        return 

    all_objects_cloud = open3d.PointCloud()
    for ix,pd in enumerate(pose_data):
        object_cls = labels[ix]
        object_pose = pd
        # object_cloud_file = osp.join(object_model_dir,object_name,"points.xyz")
        object_pose_matrix4f = get_4x4_transform(object_pose)
        # object_pose_T = object_pose[:,3]
        # object_pose_R = object_pose[:,:3]

        object_pts3d = points[object_cls] # read_xyz_file(object_cloud_file)
        pt_colors = np.zeros(object_pts3d.shape, np.float32)
        pt_colors[:] = np.array(get_random_color(), dtype=np.float32) / 255
        object_cloud = create_cloud(object_pts3d, colors=pt_colors, T=object_pose_matrix4f)
        # object_cloud.transform(object_pose_matrix4f)
        all_objects_cloud += object_cloud

        # print("Showing %s"%(object_name))
    open3d.draw_geometries([scene_cloud, all_objects_cloud])


def build_hough_voting_layer():
    from maskrcnn_benchmark.layers.hough_voting import HoughVoting

    skip_pixels = 20
    inlier_threshold = 0.9

    hv = HoughVoting(inlier_threshold=inlier_threshold, skip_pixels=skip_pixels)
    return hv

def FT(x, req_grad=False):
    return torch.tensor(x, dtype=torch.float32, device='cuda', requires_grad=req_grad)

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

    confidence_threshold = 0.95
    device = "cuda"
    cpu_device = torch.device("cpu")

    # model_file = "./checkpoints/lov_debug_pose_res14/model_final.pth"
    # image_dir = "./datasets/LOV/data"
    # image_files = ["0000/000001", "0001/000001", "0002/000001", "0003/000001", "0004/000001", "0005/000001", "0006/000001", "0007/000001", "0008/000001"]
    # image_ext = "-color.png"
    # depth_ext = "-depth.png"

    # model_file = "./checkpoints/fat_debug_res14/model_final.pth"

    # config_file = "./configs/fat_depth_debug.yaml"
    # model_file = "./checkpoints/fat_depth_debug_14/model_final.pth"
    # model_file = "./checkpoints/fat_depth_debug_14_berhu/model_final.pth"
    # image_dir = "./datasets/FAT/data"
    # image_files = ["mixed/temple_0/000885.left","mixed/temple_0/001774.left"]
    # image_ext = ".jpg"
    # depth_ext = ".depth.png"

    CLASSES = ["__background__", "class_x"]
    # config_file = "./configs/rebin.yaml"
    config_file = "./configs/loading_bbox.yaml"

    model_file = "./checkpoints/loading_bbox/model_final.pth"
    image_dir = "/home/bot/LabelMe/Images/loading_test"
    # image_files = ["mixed/temple_0/000885.left","mixed/temple_0/001774.left"]
    image_ext = ".jpg"

    cfg.merge_from_file(config_file)
    img_transformer = ImageTransformer(cfg)

    # BASE MODEL
    model = build_detection_model(cfg)
    load_model(model, model_file)
    model.eval()
    model.to(device)

    intrinsics = np.array([[768.1605834960938, 0.0, 480.0], [0.0, 768.1605834960938, 270.0], [0.0, 0.0, 1.0]])

    if cfg.MODEL.POSE_ON:
        hough_voting = build_hough_voting_layer()
        hough_voting.eval()
        hough_voting.to(device)

        points_file = "./datasets/FAT/points_all_orig.npy"
        points = np.load(points_file)

        assert len(points) == len(CLASSES)
        extents = np.zeros((len(CLASSES), 3))
        for ix in range(1,len(extents)):
            extents[ix] = np.max(points[ix], axis=0) - np.min(points[ix],axis=0)

    max_depth = 7

    for image_file in glob.glob("%s/*%s"%(image_dir, image_ext)):
    # for image_file in glob.glob("%s/*1-%s"%(image_dir, image_ext)):
    # for image_file in ["%s/%s%s"%(image_dir, f, image_ext) for f in image_files]:
        img = cv2.imread(image_file)

        # img = cv2.flip(img, 1)
        if img is None:
            print("Could not find %s"%(image_file))
            continue

        height, width = img.shape[:-1]

        # Change BGR to RGB, since torchvision.transforms use PIL image (RGB default...)
        image_list, image_scale_list = img_transformer.transform_image(img[:,:,::-1])
        image_list = image_list.to(device)

        im_scale = image_scale_list[0]

        with torch.no_grad():
            predictions = model(image_list)
        predictions = [o.to(cpu_device) for o in predictions]
        # always single image is passed at a time
        if len(predictions) == 0:
            print("No predictions for %s"%(image_file))
            continue
        predictions = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        predictions = predictions.resize((width, height))
        predictions = select_top_predictions(predictions, confidence_threshold)

        labels = predictions.get_field("labels").numpy() 
        scores = predictions.get_field("scores").numpy()
        bboxes = predictions.bbox.numpy()
        bboxes = np.round(bboxes).astype(np.int32)

        N = len(bboxes)

        if cfg.MODEL.MASK_ON:
            masks = predictions.get_field("mask").numpy().squeeze()
            label_mask = np.zeros((N, height, width), dtype=np.float32)
        if cfg.MODEL.VERTEX_ON:
            verts = predictions.get_field("vertex").numpy()
            vertex_pred = np.zeros((N, 3, height, width), dtype=np.float32)
        if cfg.MODEL.DEPTH_ON:
            pred_depths = predictions.get_field("depth").numpy()
            depth_pred = np.zeros((N, height, width), dtype=np.float32)
        if cfg.MODEL.POSE_ON:
            poses = predictions.get_field("pose").numpy()

        ix = 0
        img_copy = img.copy()
        for ix, (bbox, label, score) in enumerate(zip(bboxes, labels, scores)):

            img_copy = cv2.rectangle(img_copy, tuple(bbox[:2]), tuple(bbox[2:]), (0,0,255), 2)

            if cfg.MODEL.MASK_ON:
                mask = paste_mask_on_image(masks[ix], bbox, height, width, thresh=0.5)

                label_mask[ix] = mask

                _, contours, hierarchy = cv2.findContours(
                    mask.astype(np.uint8) * 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                color = get_random_color()
                img_copy = cv2.drawContours(img_copy, contours, -1, color, 3)
                img_copy = cv2.putText(img_copy, "%s (%.3f)"%(CLASSES[label],score), tuple(bbox[:2]), cv2.FONT_HERSHEY_COMPLEX, 0.5, color)

            if cfg.MODEL.VERTEX_ON:
                vert = verts[ix]

                cx = paste_mask_on_image(vert[0], bbox, height, width)
                cx[mask!=1] = 0
                vertex_pred[ix, 0] = cx

                cy = paste_mask_on_image(vert[1], bbox, height, width)
                cy[mask!=1] = 0
                vertex_pred[ix, 1] = cy

                # cz = paste_mask_on_image(vert[2], bbox, height, width, interp=cv2.INTER_NEAREST)
                # cz[mask!=1] = 0
                # vertex_pred[ix, 2] = cz
            
                cv2.imshow("centers x", normalize(cx, -1, 1))
                cv2.imshow("centers y", normalize(cy, -1, 1))
                # cv2.imshow("centers z", normalize(np.exp(cz), 0, 6))

            if cfg.MODEL.DEPTH_ON:
                dp = pred_depths[ix]
                mean = np.mean(dp)
                std = np.std(dp)
                dp = cv2.GaussianBlur(dp, (7,7), 0.1)
                # dp[dp <= mean - 2 * std] = 0
                m_depth = paste_mask_on_image(dp, bbox, height, width, interp=cv2.INTER_NEAREST)
                m_depth[mask != 1] = 0
                m_depth[m_depth <= mean - 2 * std] = 0
                # m_depth[mask == 1] = np.exp(m_depth[mask == 1])
                depth_pred[ix] = m_depth
                cv2.imshow("depths", normalize(m_depth, 0, max_depth))

            # cv2.imshow("masks", img_copy)
            # cv2.waitKey(0)

            ix += 1

        # cv2.imshow("masks", img_copy)
        # cv2.waitKey(0)

        if not (cfg.MODEL.DEPTH_ON or cfg.MODEL.VERTEX_ON and cfg.MODEL.POSE_ON):
            cv2.imshow("pred", img_copy)
            cv2.waitKey(0)
            continue

        factor_depth = 10000
        depth_file = image_file.replace(image_ext, depth_ext)
        depth_img = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            print("Could not find %s"%(depth_file))
            continue

        cv2.imshow("raw depth", normalize(depth_img.astype(np.float32) / factor_depth, 0, max_depth))
        cv2.waitKey(0)
            
        if cfg.MODEL.VERTEX_ON and cfg.MODEL.POSE_ON:
            vertex_pred = np.transpose(vertex_pred, [0,2,3,1])
            K = intrinsics# * im_scale  # not needed
            # K[-1,-1] = 1
            rois, final_poses = hough_voting.forward(FT(labels), FT(label_mask), FT(vertex_pred), FT(extents), FT(poses), FT(K))
            final_poses = final_poses.cpu().numpy()
            rois = rois.cpu().numpy()

            vis_rois(img, rois)
            vis_pose(img, labels, final_poses, K, points)
            cv2.waitKey(0)
            
            meta_data = {'intrinsic_matrix': K.tolist(), 'factor_depth': factor_depth}
            render_object_pose(img, depth_img, labels, meta_data, final_poses, points)

        if cfg.MODEL.DEPTH_ON:
            K = intrinsics
            meta_data = {'intrinsic_matrix': K.tolist(), 'factor_depth': factor_depth}

            render_predicted_depths(img, depth_img, depth_pred, meta_data)

        #
        # im_file_prefix = image_file.replace("color.png", "")
        # np.save("%s%s"%(im_file_prefix, "labels_mrcnn.npy"), labels)
        # np.save("%s%s"%(im_file_prefix, "masks_mrcnn.npy"), label_mask)
        # np.save("%s%s"%(im_file_prefix, "vert_pred_mrcnn.npy"), vertex_pred)
        # np.save("%s%s"%(im_file_prefix, "poses_mrcnn.npy"), poses)
        #
        # import json
        # j_file = im_file_prefix + "pred_pose.json"
        # pose_data = [{"name": CLASSES[labels[ix]], "pose": p.tolist()} for ix, p in enumerate(final_poses)]
        # with open(j_file, "w") as f:
        #     j_data = {"poses": pose_data, "meta": meta_data}
        #     json.dump(j_data, f)
        #     print("Saved pose data to %s"%(j_file))
