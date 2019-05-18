from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.data.build import build_transforms
from torchvision.transforms import functional as TVF

import numpy as np

import os
import os.path as osp

import torch
import torch.nn.functional as F

import cv2


def load_model(model, f):
    from maskrcnn_benchmark.utils.model_serialization import load_state_dict

    checkpoint = torch.load(f, map_location=torch.device("cpu"))
    load_state_dict(model, checkpoint.pop("model"))
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
        # image = image.unsqueeze(0)
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        return image_list, [target.get_field('scale')]


def select_top_predictions(predictions, confidence_threshold=0.7, score_field="scores"):
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
    scores = predictions.get_field(score_field)
    keep = torch.nonzero(scores > confidence_threshold).squeeze(1)
    predictions = predictions[keep]
    scores = predictions.get_field(score_field)
    _, idx = scores.sort(0, descending=True)
    return predictions[idx]


def paste_mask_on_image(mask, box, im_h, im_w, thresh=None, interp=cv2.INTER_LINEAR, rotated=False):

    if rotated:
        assert len(box) == 5  # xc,yc,w,h,angle
        w = box[2]
        h = box[3]
    else:
        assert len(box) == 4  # x1,y1,x2,y2
        w = box[2] - box[0] + 1
        h = box[3] - box[1] + 1

    w = max(w, 1)
    h = max(h, 1)

    w = int(np.round(w))
    h = int(np.round(h))

    resized = cv2.resize(mask, (w, h), interpolation=interp)

    if thresh is not None:#thresh >= 0:
        resized = (resized > thresh).astype(np.float32)

    canvas = np.zeros((im_h, im_w), dtype=np.float32)

    if rotated:
        from maskrcnn_benchmark.modeling.rotate_ops import paste_rotated_roi_in_image

        canvas = paste_rotated_roi_in_image(canvas, resized, box)

    else:
        x_0 = max(box[0], 0)
        x_1 = min(box[2] + 1, im_w)
        y_0 = max(box[1], 0)
        y_1 = min(box[3] + 1, im_h)

        canvas[y_0:y_1, x_0:x_1] = resized[(y_0 - box[1]) : (y_1 - box[1]), (x_0 - box[0]) : (x_1 - box[0])]
    # cv2.imshow("canvas", canvas)
    # cv2.waitKey(0)
    return canvas


def get_random_color():
    return (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))


if __name__ == '__main__':
    import glob

    confidence_threshold = 0.95
    device = "cuda"
    cpu_device = torch.device("cpu")

    CLASSES = ["__background__", "class_x"]
    config_file = "./configs/coco_rotated_rpn_only.yaml"
    model_file = "./checkpoints/coco_rotated_rpn_only/model_final.pth"

    # config_file = "./configs/coco_rotated_faster_rcnn.yaml"
    # model_file = "./checkpoints/coco_rotated_faster_rcnn_debug/model_final.pth"
    config_file = "./configs/coco_rotated_mask_rcnn_fpn.yaml"
    model_file = "./checkpoints/coco_rotated_mask_fpn/model_final.pth"

    image_dir = "/data/MSCOCO/train2014"
    # image_files = ["mixed/temple_0/000885.left","mixed/temple_0/001774.left"]
    image_ext = ".jpg"
    # image_files = [u'COCO_val2014_000000001000.jpg', u'COCO_val2014_000000010012.jpg']
    image_files = ['COCO_val2014_000000415360.jpg',
         'COCO_val2014_000000438915.jpg',
         'COCO_val2014_000000209028.jpg',
         'COCO_val2014_000000500100.jpg']#[1:]
    image_files = [
        u'COCO_train2014_000000417793.jpg',
        u'COCO_train2014_000000147459.jpg',
        u'COCO_train2014_000000417797.jpg',
        u'COCO_train2014_000000032778.jpg',
        u'COCO_train2014_000000393227.jpg',
        u'COCO_train2014_000000139276.jpg',
        u'COCO_train2014_000000114703.jpg',
        u'COCO_train2014_000000229398.jpg',
        u'COCO_train2014_000000401435.jpg',
        u'COCO_train2014_000000581667.jpg',
        u'COCO_train2014_000000213034.jpg',
        u'COCO_train2014_000000024621.jpg',
        u'COCO_train2014_000000294962.jpg',
        u'COCO_train2014_000000548926.jpg',
        u'COCO_train2014_000000188482.jpg',
        u'COCO_train2014_000000337707.jpg',
        u'COCO_train2014_000000458827.jpg',
        u'COCO_train2014_000000000077.jpg',
        u'COCO_train2014_000000417870.jpg',
        u'COCO_train2014_000000163921.jpg',
        u'COCO_train2014_000000516184.jpg',
        u'COCO_train2014_000000237658.jpg',
        u'COCO_train2014_000000204891.jpg',
        u'COCO_train2014_000000467038.jpg',
        u'COCO_train2014_000000270440.jpg',
        u'COCO_train2014_000000311401.jpg',
        u'COCO_train2014_000000221293.jpg',
        u'COCO_train2014_000000565361.jpg',
        u'COCO_train2014_000000139380.jpg',
        u'COCO_train2014_000000019134.jpg'
    ][::-1]
    image_files = [osp.join(image_dir, f) for f in image_files]

    # config_file = "./configs/coco_rotated_mask_rcnn_fpn.yaml"
    # model_file = "./checkpoints/coco_rotated_loading_mask_fpn2/model_final.pth"
    # config_file = "./configs/coco_mask_rcnn.yaml"
    # model_file = "./checkpoints/coco_loading_mask/model_final.pth"
    # image_dir = "/home/bot/LabelMe/Images/loading_test"
    # image_files = glob.glob("%s/*.jpg"%(image_dir))[::-1]

    cfg.merge_from_file(config_file)
    img_transformer = ImageTransformer(cfg)

    # BASE MODEL
    model = build_detection_model(cfg)
    load_model(model, model_file)
    model.eval()
    model.to(device)

    for image_file in image_files: #glob.glob("%s/*%s"%(image_dir, image_ext)):
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

        score_field = "scores"
        if cfg.MODEL.RPN_ONLY:
            score_field = "objectness"
        # reshape prediction (a BoxList) into the original image size
        pred_size = predictions.size
        predictions = predictions.resize((width, height))
        predictions = select_top_predictions(predictions, confidence_threshold, score_field)

        bboxes = predictions.bbox.numpy()
        bboxes = np.round(bboxes).astype(np.int32)

        N = len(bboxes)
        img_copy = img.copy()

        scores = predictions.get_field(score_field).numpy()  # from roi box head

        if not cfg.MODEL.RPN_ONLY:
            labels = predictions.get_field("labels").numpy()
        if cfg.MODEL.MASK_ON:
            masks = predictions.get_field("mask").numpy().squeeze(1)
            label_mask = np.zeros((N, height, width), dtype=np.float32)
        if cfg.MODEL.ROTATED:
            from maskrcnn_benchmark.modeling.rrpn.anchor_generator import draw_anchors
            rrects = predictions.get_field("rrects").rbox.cpu().numpy()

        for ix, (bbox, score) in enumerate(zip(bboxes, scores)):

            if not cfg.MODEL.RPN_ONLY:
                label = labels[ix]

            if 1:#not cfg.MODEL.MASK_ON:  # if no masks, draw bboxes
                if cfg.MODEL.ROTATED:

                    rr = rrects[ix]
                    img_copy = draw_anchors(img_copy, [rr], [[0,0,255]])
                else:
                    img_copy = cv2.rectangle(img_copy, tuple(bbox[:2]), tuple(bbox[2:]), (0, 0, 255), 2)
            if cfg.MODEL.MASK_ON:  # if no masks, draw bboxes
                rotated = cfg.MODEL.ROTATED
                rbox = bbox if not rotated else rrects[ix]
                if not cfg.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS:
                    mask = paste_mask_on_image(masks[ix], rbox, height, width, thresh=0.5, rotated=rotated)
                else:
                    mask = cv2.resize(masks[ix], (width, height))

                label_mask[ix] = mask

                _, contours, hierarchy = cv2.findContours(
                    mask.astype(np.uint8) * 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                color = get_random_color()
                img_copy = cv2.drawContours(img_copy, contours, -1, color, 3)
                # img_copy = cv2.putText(img_copy, "%s (%.3f)"%(CLASSES[label],score), tuple(bbox[:2]), cv2.FONT_HERSHEY_COMPLEX, 0.5, color)

        cv2.imshow("img", img)
        cv2.imshow("pred", img_copy)
        cv2.waitKey(0)
