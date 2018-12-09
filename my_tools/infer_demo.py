from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.layers.hough_voting import HoughVoting

from torchvision import transforms as T

import numpy as np
from transforms3d.quaternions import quat2mat#, mat2quat

import torch
import torch.nn.functional as F

import glob
import cv2

def load_model(model, f):
    checkpoint = torch.load(f, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint.pop("model"))
    print("Loaded %s"%(f))


class ImageTransformer(object):
    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.transforms = self.build_transform()

    def build_transform(self):
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(cfg.INPUT.MIN_SIZE_TEST),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def transform_image(self, original_image):
        image = self.transforms(original_image)
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image = image.unsqueeze(0)
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        return image_list

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

def paste_mask_on_image(mask, box, im_h, im_w, thresh=None):
    w = box[2] - box[0] + 1
    h = box[3] - box[1] + 1
    w = max(w, 1)
    h = max(h, 1)

    resized = cv2.resize(mask, (w,h), interpolation=cv2.INTER_LINEAR)

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

def vis_pose(im, labels, poses, intrinsics, points):

    img = im.copy()

    N = poses.shape[0]

    colors = [get_random_color() for i in range(N)]
    for i in range(N):
        cls = labels[i]
        if cls > 0:
            # extract 3D points
            x3d = np.ones((4, points.shape[1]), dtype=np.float32)
            x3d[0, :] = points[cls,:,0]
            x3d[1, :] = points[cls,:,1]
            x3d[2, :] = points[cls,:,2]

            # projection
            RT = np.zeros((3, 4), dtype=np.float32)
            RT[:3, :3] = quat2mat(poses[i, :4])
            RT[:, 3] = poses[i, 4:7]
            x2d = np.matmul(intrinsics, np.matmul(RT, x3d))
            x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
            x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])

            color = colors[i]

            proj_pts = x2d[:2].transpose()
            proj_pts = np.round(proj_pts).astype(np.int32)
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

def build_hough_voting_layer():
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
    config_file = "./configs/lov_debug.yaml"
    device = "cuda"
    cpu_device = torch.device("cpu")

    # update the config options with the config file
    cfg.merge_from_file(config_file)

    img_transformer = ImageTransformer(cfg)

    model_file = "./checkpoints/lov_debug_pose_res14/model_final.pth"
    # model_file = "./checkpoints/lov_debug_res14/model_0001000.pth"

    # BASE MODEL
    model = build_detection_model(cfg)
    load_model(model, model_file)
    model.eval()
    model.to(device)

    hough_voting = build_hough_voting_layer()
    hough_voting.eval()
    hough_voting.to(device)

    intrinsics = np.array([[1066.778, 0, 312.9869], [0, 1067.487, 241.3109], [0.0, 0.0, 1.0]])
    # T_intrinsics = FT(intrinsics)

    image_dir = "./datasets/LOV/data"
    points_file = "./datasets/LOV/points_all_orig.npy"  # TODO: CHANGE
    points = np.load(points_file)

    assert len(points) == len(CLASSES)
    extents = np.max(points, axis=1) - np.min(points,axis=1)

    image_files = ["0000/000001", "0001/000001", "0002/000001", "0003/000001", "0004/000001", "0005/000001", "0006/000001", "0007/000001", "0008/000001"]
    image_ext = "color.png"
    # for image_file in glob.glob("%s/*1-%s"%(image_dir, image_ext)):
    for image_file in ["%s/%s-%s"%(image_dir, f, image_ext) for f in image_files]:
        img = cv2.imread(image_file)
        # img = cv2.flip(img, 1)
        if img is None:
            print("Could not find %s"%(image_file))
            continue

        height, width = img.shape[:-1]

        image_list = img_transformer.transform_image(img)
        image_list = image_list.to(device)

        im_scale = 1.0  # TODO: Get im scale from image transformer
        K = intrinsics * im_scale

        with torch.no_grad():
            predictions = model(image_list)
        predictions = [o.to(cpu_device) for o in predictions]
        # always single image is passed at a time
        predictions = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        predictions = predictions.resize((width, height))
        predictions = select_top_predictions(predictions, confidence_threshold)

        labels = predictions.get_field("labels").numpy() 
        verts = predictions.get_field("vertex").numpy()
        masks = predictions.get_field("mask").numpy().squeeze()
        poses = predictions.get_field("pose").numpy()
        scores = predictions.get_field("scores").numpy()
        bboxes = predictions.bbox.numpy()
        bboxes = np.round(bboxes).astype(np.int32)

        canvas = np.zeros((height, width), dtype=np.float32)
        thresh = 0.5

        N = len(masks)
        label_mask = np.zeros((N, height, width), dtype=np.float32)
        vertex_pred = np.zeros((N, 3, height, width), dtype=np.float32)

        ix = 0
        img_copy = img.copy()
        for bbox, mask, vert, label, score in zip(bboxes, masks, verts, labels, scores):

            mask = paste_mask_on_image(mask, bbox, height, width, thresh=thresh)
            cv2.imshow("mask", mask)

            label_mask[ix] = mask

            _, contours, hierarchy = cv2.findContours(
                mask.astype(np.uint8) * 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            color = get_random_color()
            img_copy = cv2.drawContours(img_copy, contours, -1, color, 3)
            img_copy = cv2.putText(img_copy, "%s (%.3f)"%(CLASSES[label],score), tuple(bbox[:2]), cv2.FONT_HERSHEY_COMPLEX, 0.5, color)

            cx = paste_mask_on_image(vert[0], bbox, height, width)
            cx[mask!=1] = 0
            vertex_pred[ix, 0] = cx
            cv2.imshow("centers x", normalize(cx, -1, 1))

            cy = paste_mask_on_image(vert[1], bbox, height, width)
            cy[mask!=1] = 0
            vertex_pred[ix, 1] = cy
            cv2.imshow("centers y", normalize(cy, -1, 1))

            cz = paste_mask_on_image(vert[2], bbox, height, width)
            cz[mask!=1] = 0
            vertex_pred[ix, 2] = cz
            cv2.imshow("centers z", normalize(np.exp(cz), 0, 6))

            cv2.imshow("masks", img_copy)
            cv2.waitKey(0)

            ix += 1

        vertex_pred = np.transpose(vertex_pred, [0,2,3,1])
        rois, final_poses = hough_voting.forward(FT(labels), FT(label_mask), FT(vertex_pred), FT(extents), FT(poses), FT(K.flatten()))
        final_poses = final_poses.cpu().numpy()
        rois = rois.cpu().numpy()

        vis_rois(img, rois)
        vis_pose(img, labels, final_poses, K, points)
        cv2.waitKey(0)

        # #     # np.stack(())
        # label_mask = np.expand_dims(label_mask, axis=0)
        # vertex_pred = np.expand_dims(vertex_pred, axis=0)
        np.save(image_file.replace("color.png", "labels_mrcnn.npy"), labels)
        np.save(image_file.replace("color.png", "masks_mrcnn.npy"), label_mask)
        np.save(image_file.replace("color.png", "vert_pred_mrcnn.npy"), vertex_pred)
        np.save(image_file.replace("color.png", "poses_mrcnn.npy"), poses)

