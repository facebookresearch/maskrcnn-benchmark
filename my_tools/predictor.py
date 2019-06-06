import numpy as np
import cv2

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TVF

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.data.build import build_transforms

from maskrcnn_benchmark.utils.model_serialization import load_state_dict


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


def load_model(model, f):
    checkpoint = torch.load(f, map_location=torch.device("cpu"))
    load_state_dict(model, checkpoint.pop("model"))
    print("Loaded %s"%(f))


class Predictor(object):
    def __init__(self, config_file, min_score=0.8, device="cuda"):
        
        cfg.merge_from_file(config_file)

        self.cfg = cfg
        self.min_score = min_score

        self.device = device
        self.cpu_device = torch.device("cpu")

        self.model = self.build_model()
        self.img_transformer = ImageTransformer(self.cfg)

        self.cnt = 0

        self.score_field = "scores"
        if cfg.MODEL.RPN_ONLY:
            self.score_field = "objectness"
        elif cfg.MODEL.MASKIOU_ON:
            self.score_field = "mask_scores"

    def get_data_from_prediction(self, predictions, img_height, img_width):
        data = {
            "scores": predictions.get_field(self.score_field).numpy(),  # from roi box head,
            "bboxes": predictions.bbox.numpy()
        }
        if not self.cfg.MODEL.RPN_ONLY:
            data["labels"] = predictions.get_field("labels").numpy()

        rotated = self.cfg.MODEL.ROTATED
        if rotated:
            data["rrects"] = predictions.get_field("rrects").rbox.cpu().numpy()

        if self.cfg.MODEL.MASK_ON:
            masks = predictions.get_field("mask").numpy().squeeze(1)
            N = len(masks)

            boxes = data["bboxes"] if not rotated else data["rrects"]
            assert N == len(boxes)

            is_pp_mask = self.cfg.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS

            final_masks = np.zeros((N, img_height, img_width), dtype=np.float32)
            for ix in range(N):
                box = boxes[ix]
                if not is_pp_mask:
                    mask = paste_mask_on_image(masks[ix], box, img_height, img_width, thresh=0.5, rotated=rotated)
                else:
                    mask = cv2.resize(masks[ix], (img_width, img_height))

                final_masks[ix] = mask

            data["masks"] = final_masks

        return data

    def run_on_opencv_image(self, img):
        # assert len(img.shape) == 3 and img.shape[-1] == 3  # image must be in (H,W,3) dims
        height, width, cn = img.shape

        if self.cnt == 0:
            self.model.to(self.device)

        self.cnt += 1

        # Change BGR to RGB, since torchvision.transforms use PIL image (RGB default...)
        image_list, image_scale_list = self.img_transformer.transform_image(img[:,:,::-1])
        image_list = image_list.to(self.device)

        with torch.no_grad():
            predictions = self.model(image_list)
        predictions = [o.to(self.cpu_device) for o in predictions]

        if len(predictions) == 0:
            return None

        predictions = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        predictions = predictions.resize((width, height))
        predictions = select_top_predictions(predictions, self.min_score, self.score_field)

        data = self.get_data_from_prediction(predictions, height, width)

        return data

    def build_model(self):
        # BASE MODEL
        model = build_detection_model(self.cfg)
        model.eval()
        return model

    def load_weights(self, model_file):
        load_model(self.model, model_file)
