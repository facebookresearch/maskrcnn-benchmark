import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers.ave_dist_loss import AverageDistanceLoss
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.matcher import Matcher

from maskrcnn_benchmark.modeling.utils import cat

class PoseRCNNLossComputation(object):

    def __init__(self, proposal_matcher):
        """
        Arguments:
            proposal_matcher (Matcher)
        """
        self.proposal_matcher = proposal_matcher
        self.loss = AverageDistanceLoss(margin=0.01)
        # self.batch_size_per_image = batch_size_per_image

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        #  RCNN needs "labels" and "poses "fields for creating the targets
        target = target.copy_with_fields(["labels", "poses"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        pose_targets = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # this can probably be removed, but is left here for clarity
            # and completeness
            neg_inds = matched_idxs < 0 # Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0

            poses = matched_targets.get_field("poses")[:,:4]  # get q from pose field (qw,qx,qy,qz,x,y,z)

            labels.append(labels_per_image)
            pose_targets.append(poses)

        return labels, pose_targets

    def __call__(self, proposals, pose_pred, targets):
        """
        Arguments:
            proposals (list[BoxList])
            pose_pred (Tensor)
            targets (list[BoxList])

        Return:
            pose_loss (Tensor): scalar tensor containing the loss
        """
        labels, pose_targets = self.prepare_targets(proposals, targets)

        labels = cat(labels, dim=0)
        pose_targets = cat(pose_targets, dim=0)

        # only computed on positive samples
        positive_inds = torch.nonzero(labels > 0).squeeze(1)
        pose_targets = pose_targets[positive_inds]
        labels_pos = labels[positive_inds]

        if pose_targets.numel() == 0:
            return pose_pred.sum() * 0

        vp_size = pose_pred.shape
        N,C = vp_size
        pp = pose_pred.view(N, -1, 4)  # N,classes,4
        pp = pp[positive_inds, labels_pos]  # N,4

        points = targets[0].get_field("points")
        symmetry = targets[0].get_field("symmetry")

        # import numpy as np
        # np.save("poses_preds.npy", pp.detach().cpu().numpy())
        # np.save("poses_targets.npy", pose_targets.detach().cpu().numpy())
        # np.save("poses_labels.npy", labels_pos.detach().cpu().numpy())
        # np.save("points.npy", points.cpu().numpy())
        # np.save("symmetry.npy", symmetry.cpu().numpy())

        POSE_W = 1.0
        pose_loss = self.loss(pp, pose_targets, labels_pos, points, symmetry) * POSE_W
        return pose_loss


def make_roi_pose_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_POSE_HEAD.FG_IOU_THRESHOLD,
        0,
        allow_low_quality_matches=False,
    )

    loss_evaluator = PoseRCNNLossComputation(matcher)

    return loss_evaluator
