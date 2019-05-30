import numpy as np
import torch
# from torch import nn

from maskrcnn_benchmark.layers.misc import interpolate

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import MaskPostProcessor, expand_masks
from maskrcnn_benchmark.modeling.rotate_ops import paste_rotated_roi_in_image
from maskrcnn_benchmark.structures.rotated_box import RotatedBox


def paste_mask_in_image(mask, box, im_h, im_w, thresh=0.5, padding=1):
    # Need to work on the CPU, where fp16 isn't supported - cast to float to avoid this
    mask = mask.float()
    box = box.float()

    assert len(box) == 5  # xc yc w h angle

    padded_mask, scale = expand_masks(mask[None], padding=padding)
    mask = padded_mask
    # box = expand_boxes(box[None], scale)[0]
    box[2:4] *= scale

    w = int(np.round(box[2]))
    h = int(np.round(box[3]))
    w = max(w, 1)
    h = max(h, 1)

    # Set shape to [batchxCxHxW]
    mask = mask.expand((1, 1, -1, -1))

    # Resize mask
    mask = mask.to(torch.float32)
    mask = interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
    mask = mask[0][0]

    if thresh >= 0:
        mask = (mask > thresh).to(dtype=torch.float32)

    im_mask = np.zeros((im_h, im_w), dtype=np.float32)
    im_mask = paste_rotated_roi_in_image(im_mask, mask.cpu().numpy(), box)

    return torch.from_numpy(im_mask).to(mask.device)


def paste_masks_in_image(masks, box, im_h, im_w, thresh=0.5, padding=1):
    N = len(masks)
    canvas = torch.zeros((N, im_h, im_w), dtype=torch.float32)
    box = box.cpu()
    for ix,mask in enumerate(masks.cpu()):
        canvas[ix] = paste_mask_in_image(mask, box, im_h, im_w, thresh, padding)
    return canvas


class Masker(object):
    """
    Projects a set of masks in an image on the locations
    specified by the bounding boxes
    """

    def __init__(self, threshold=0.5, padding=1):
        self.threshold = threshold
        self.padding = padding

    def forward_single_image(self, masks, boxes):
        # boxes = boxes.convert("xyxy")
        im_w, im_h = boxes.size
        rr = boxes.get_field('rrects')
        rrects = rr.rbox if isinstance(rr, RotatedBox) else rr
        res = [
            paste_masks_in_image(mask, box, im_h, im_w, self.threshold, self.padding)
            for mask, box in zip(masks, rrects)
        ]
        if len(res) > 0:
            res = torch.stack(res, dim=0)#[:, None]
        else:
            res = masks.new_empty((0, 1, masks.shape[-2], masks.shape[-1]))
        return res

    def __call__(self, masks, boxes):
        if isinstance(boxes, BoxList):
            boxes = [boxes]

        # Make some sanity check
        assert len(boxes) == len(masks), "Masks and boxes should have the same length."

        # TODO:  Is this JIT compatible?
        # If not we should make it compatible.
        results = []
        for mask, box in zip(masks, boxes):
            assert mask.shape[0] == len(box), "Number of objects should be the same."
            result = self.forward_single_image(mask, box)
            results.append(result)
        return results


class MaskPostProcessor2(MaskPostProcessor):
    """
    MaskPostProcessor with extra classifier handling
    """

    def __init__(self, masker=None):
        super(MaskPostProcessor2, self).__init__(masker)

    def forward(self, x, boxes, cls_logits=None):
        """
        Arguments:
            x (Tensor): the mask logits
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image
            cls_logits (Tensor): the mask branch classification logits

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra field mask
        """
        if cls_logits is not None:
            cls_prob = cls_logits.sigmoid()
            # filter cls_prob below 0.5

        results = super(MaskPostProcessor2, self).forward(x, boxes)
        return results


def make_roi_mask_post_processor(cfg):
    if cfg.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS:
        mask_threshold = cfg.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS_THRESHOLD
        masker = Masker(threshold=mask_threshold, padding=1)
    else:
        masker = None
    mask_post_processor = MaskPostProcessor(masker)
    return mask_post_processor
