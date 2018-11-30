import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from maskrcnn_benchmark.structures.bounding_box import BoxList

class VertexPostProcessor(nn.Module):
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.

    If a masker object is passed, it will additionally
    project the masks in the image according to the locations in boxes,
    """

    def __init__(self, masker=None):
        super(VertexPostProcessor, self).__init__()
        self.masker = masker

    def forward(self, vert_pred, boxes):
        """
        Arguments:
            vert_pred (Tensor): the vertex pred values  (num_classes * 3)
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra field mask
        """
        # mask_prob = x.sigmoid()

        # select masks coresponding to the predicted classes
        num_masks = vert_pred.shape[0]
        labels = [bbox.get_field("labels") for bbox in boxes]
        labels = torch.cat(labels)
        index = torch.arange(num_masks, device=labels.device)

        # vert_pred = vert_pred[index, labels][:, None]  #
        N, C, H, W = vert_pred.shape
        vert_pred = vert_pred.view(N,-1,3,H,W)
        vert_pred = vert_pred[index, labels]

        boxes_per_image = [len(box) for box in boxes]
        vert_pred = vert_pred.split(boxes_per_image, dim=0)

        if self.masker:
            vert_pred = self.masker(vert_pred, boxes)

        results = []
        for vp, box in zip(vert_pred, boxes):
            bbox = BoxList(box.bbox, box.size, mode="xyxy")
            for field in box.fields():
                bbox.add_field(field, box.get_field(field))
            bbox.add_field("vertex", vp)
            results.append(bbox)

        return results


def make_roi_vertex_post_processor(cfg):
    # if cfg.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS:
    #     mask_threshold = cfg.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS_THRESHOLD
    #     masker = Vertexer(threshold=mask_threshold, padding=1)
    # else:
    #     masker = None
    masker = None
    mask_post_processor = VertexPostProcessor(masker)
    return mask_post_processor
