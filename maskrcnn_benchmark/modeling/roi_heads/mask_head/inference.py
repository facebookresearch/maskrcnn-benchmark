# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from maskrcnn_benchmark.structures.bounding_box import BoxList


# TODO check if want to return a single BoxList or a composite
# object
class MaskPostProcessor(nn.Module):
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.

    If a masker object is passed, it will additionally
    project the masks in the image according to the locations in boxes,
    """

    def __init__(self, masker=None):
        super(MaskPostProcessor, self).__init__()
        self.masker = masker

    def forward(self, x, boxes):
        """
        Arguments:
            x (Tensor): the mask logits
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra field mask
        """
        mask_prob = x.sigmoid()

        # select masks coresponding to the predicted classes
        num_masks = x.shape[0]
        labels = [bbox.get_field("labels") for bbox in boxes]
        labels = torch.cat(labels)
        index = torch.arange(num_masks, device=labels.device)
        mask_prob = mask_prob[index, labels][:, None]

        if self.masker:
            mask_prob = self.masker(mask_prob, boxes)

        boxes_per_image = [len(box) for box in boxes]
        mask_prob = mask_prob.split(boxes_per_image, dim=0)

        results = []
        for prob, box in zip(mask_prob, boxes):
            bbox = BoxList(box.bbox, box.size, mode="xyxy")
            for field in box.fields():
                bbox.add_field(field, box.get_field(field))
            bbox.add_field("mask", prob)
            results.append(bbox)

        return results


class MaskPostProcessorCOCOFormat(MaskPostProcessor):
    """
    From the results of the CNN, post process the results
    so that the masks are pasted in the image, and
    additionally convert the results to COCO format.
    """

    def forward(self, x, boxes):
        import pycocotools.mask as mask_util
        import numpy as np

        results = super(MaskPostProcessorCOCOFormat, self).forward(x, boxes)
        for result in results:
            masks = result.get_field("mask").cpu()
            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")
            result.add_field("mask", rles)
        return results


# the next two functions should be merged inside Masker
# but are kept here for the moment while we need them
# temporarily gor paste_mask_in_image
def expand_boxes(boxes, scale):
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    boxes_exp = torch.zeros_like(boxes)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half
    return boxes_exp


def expand_masks(mask, padding):
    N = mask.shape[0]
    M = mask.shape[-1]
    pad2 = 2 * padding
    scale = float(M + pad2) / M
    padded_mask = mask.new_zeros((N, 1, M + pad2, M + pad2))
    padded_mask[:, :, padding:-padding, padding:-padding] = mask
    return padded_mask, scale


class Masker(object):
    """
    Projects a set of masks in an image on the locations
    specified by the bounding boxes
    """

    def __init__(self, threshold=0.5, padding=1):
        self.threshold = threshold
        self.padding = padding

    # TODO this gives slightly different results
    # than the Detectron implementation. Fix it
    def compute_flow_field_cpu(self, boxes):
        im_w, im_h = boxes.size
        boxes_data = boxes.bbox
        num_boxes = len(boxes_data)
        device = boxes_data.device

        TO_REMOVE = 1
        boxes_data = boxes_data.int()
        box_widths = boxes_data[:, 2] - boxes_data[:, 0] + TO_REMOVE
        box_heights = boxes_data[:, 3] - boxes_data[:, 1] + TO_REMOVE

        box_widths.clamp_(min=1)
        box_heights.clamp_(min=1)

        boxes_data = boxes_data.tolist()
        box_widths = box_widths.tolist()
        box_heights = box_heights.tolist()

        flow_field = torch.full((num_boxes, im_h, im_w, 2), -2)

        for i in range(num_boxes):
            w = box_widths[i]
            h = box_heights[i]
            if w < 2 or h < 2:
                continue
            x = torch.linspace(-1, 1, w)
            y = torch.linspace(-1, 1, h)
            # meshogrid
            x = x[None, :].expand(h, w)
            y = y[:, None].expand(h, w)

            b = boxes_data[i]
            x_0 = max(b[0], 0)
            x_1 = min(b[2] + 0, im_w)
            y_0 = max(b[1], 0)
            y_1 = min(b[3] + 0, im_h)
            flow_field[i, y_0:y_1, x_0:x_1, 0] = x[
                (y_0 - b[1]) : (y_1 - b[1]), (x_0 - b[0]) : (x_1 - b[0])
            ]
            flow_field[i, y_0:y_1, x_0:x_1, 1] = y[
                (y_0 - b[1]) : (y_1 - b[1]), (x_0 - b[0]) : (x_1 - b[0])
            ]

        return flow_field.to(device)

    def compute_flow_field_gpu(self, boxes):
        from torch_detectron import layers

        width, height = boxes.size
        flow_field = layers.compute_flow(boxes.bbox, height, width)
        return flow_field

    def compute_flow_field(self, boxes):
        if boxes.bbox.is_cuda:
            return self.compute_flow_field_gpu(boxes)
        return self.compute_flow_field_cpu(boxes)

    # TODO make it work better for batches
    def forward_single_image(self, masks, boxes):
        boxes = boxes.convert("xyxy")
        if self.padding:
            boxes = BoxList(boxes.bbox.clone(), boxes.size, boxes.mode)
            masks, scale = expand_masks(masks, self.padding)
            boxes.bbox = expand_boxes(boxes.bbox, scale)

        flow_field = self.compute_flow_field(boxes)
        masks = masks.to(torch.float32)
        result = torch.nn.functional.grid_sample(masks, flow_field)
        if self.threshold > 0:
            result = result > self.threshold
        return result

    def __call__(self, masks, boxes:
        # TODO do this properly
        if isinstance(boxes, BoxList):
            boxes = [boxes]
        assert len(boxes) == 1, "Only single image batch supported"
        result = self.forward_single_image(masks, boxes[0])
        return result


def make_roi_mask_post_processor(cfg):
    masker = None
    mask_post_processor = MaskPostProcessor(masker)
    return mask_post_processor
