import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from maskrcnn_benchmark.structures.bounding_box import BoxList


class PosePostProcessor(nn.Module):
    """
    """

    def __init__(self):
        super(PosePostProcessor, self).__init__()

    def forward(self, pose_pred, boxes):
        """
        Arguments:
            pose_pred (Tensor): the pose pred values  (N, num_classes * 4)
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra field 'pose'
        """

        # select pose corresponding to the predicted classes
        N, C = pose_pred.shape
        labels = [bbox.get_field("labels") for bbox in boxes]
        labels = torch.cat(labels)
        index = torch.arange(N, device=labels.device)

        results = []
        if pose_pred.numel() == 0:
            return results

        pose_pred = pose_pred.view(N,-1,5)  # N,Classes,4
        pose_pred = pose_pred[index, labels]  # N,4

        boxes_per_image = [len(box) for box in boxes]
        pose_pred = pose_pred.split(boxes_per_image, dim=0)

        for pp, box in zip(pose_pred, boxes):
            bbox = BoxList(box.bbox, box.size, mode="xyxy")
            for field in box.fields():
                bbox.add_field(field, box.get_field(field))
            bbox.add_field("pose", pp)
            results.append(bbox)

        return results


def make_roi_pose_post_processor(cfg):

    mask_post_processor = PosePostProcessor()
    return mask_post_processor
