import torch
from torch.nn import functional as F

from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.losses.target_preparator import TargetPreparator
from maskrcnn_benchmark.modeling.utils import cat, cat_bbox, split_with_sizes, keep_only_positive_boxes
from maskrcnn_benchmark.layers import smooth_l1_loss

from maskrcnn_benchmark.structures.keypoint import keypoints_to_heat_map


class KeypointTargetPreparator(TargetPreparator):
    def __init__(self, proposal_matcher, discretization_size):
        super(KeypointTargetPreparator, self).__init__(proposal_matcher, None)
        self.discretization_size = discretization_size

    def index_target(self, target, index):
        target = target.copy_with_fields(['labels', 'keypoints'])
        return target[index]

    def prepare_labels(self, matched_targets_per_image, anchors_per_image):
        matched_idxs = matched_targets_per_image.get_field('matched_idxs')
        labels_per_image = matched_targets_per_image.get_field('labels')
        labels_per_image = labels_per_image.to(dtype=torch.int64)

        keypoints = matched_targets_per_image.get_field('keypoints')
        # TODO remove conditional  when better support for zero-dim is in
        if keypoints.keypoints.numel() > 0:
            within_box = self._within_box(keypoints.keypoints, matched_targets_per_image.bbox)
            vis_kp = keypoints.keypoints[..., 2] > 0
            is_visible = (within_box & vis_kp).sum(1) > 0

            labels_per_image[~is_visible] = -1

        # this can probably be removed, but is left here for clarity
        # and completeness
        neg_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
        labels_per_image[neg_inds] = 0

        return labels_per_image, keypoints

    def _within_box(self, points, boxes):
        """Validate which keypoints are contained inside a given box.
        points: NxKx2
        boxes: Nx4
        output: NxK
        """
        x_within = (points[..., 0] >= boxes[:, 0, None]) & (points[..., 0] <= boxes[:, 2, None])
        y_within = (points[..., 1] >= boxes[:, 1, None]) & (points[..., 1] <= boxes[:, 3, None])
        return x_within & y_within

    def project(self, keypoints, anchors):
        """
        """
        M = self.discretization_size
        device = anchors.bbox.device
        positive_anchors = anchors.convert('xyxy')
        # assert keypoints.size == anchors.size, '{}, {}'.format(
        #         keypoints, anchors)

        # return keypoints_to_heat_map(keypoints.keypoints, anchors.bbox, M)
        return keypoints_to_heat_map(keypoints, anchors.bbox, M)


class KeypointRCNNLossComputation(object):
    def __init__(self, target_preparator, fg_bg_sampler):
        """
        """
        self.target_preparator = target_preparator
        self.fg_bg_sampler = fg_bg_sampler


    # TODO remove as it's almost a copy-paste from fast_rcnn_losses
    def subsample(self, anchors, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled anchors.
        Note: this function keeps a state.
        Arguments:
            anchors (list of list of BoxList)
            targets (list of BoxList)
        """

        labels, keypoints = self.prepare_targets(anchors, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        # FIXME hack to simplify a few things.
        sampled_neg_inds = [i.new() for i in sampled_neg_inds]

        # flip anchors to be images -> feature map levels
        anchors = list(zip(*anchors))
        levels = [torch.tensor([i for i, n in enumerate(anchor)
            for _ in range(n.bbox.shape[0])]) for anchor in anchors]
        num_levels = len(anchors[0])
        num_images = len(anchors)
        # concatenate all anchors for the same image
        anchors = [cat_bbox(anchors_per_image) for anchors_per_image in anchors]

        # add corresponding label information to the bounding boxes
        # this can be used with `keep_only_positive_boxes` in order to
        # restrict the set of boxes to be used during other steps (Mask R-CNN
        # for example)
        for labels_per_image, keypoints_per_image, anchors_per_image in zip(
                labels, keypoints, anchors):
            anchors_per_image.add_field('labels', labels_per_image)
            anchors_per_image.add_field('keypoints', keypoints_per_image)

        sampled_inds = []
        sampled_image_levels = []
        # distributed sampled anchors, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
                zip(sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = pos_inds_img  # | neg_inds_img
            anchors_per_image = anchors[img_idx][img_sampled_inds]
            sampled_levels = levels[img_idx][img_sampled_inds]
            # keypoints_per_image = keypoints[img_idx][img_sampled_inds]
            # TODO replace with bincount because indices in the same level
            # are packed together
            anchors_per_level_per_image = []
            sampled_image_level_temp = []
            # keypoints_per_level_per_image = []
            for level in range(num_levels):
                level_idx = nonzero(sampled_levels == level)[0]
                anchors_per_level_per_image.append(anchors_per_image[level_idx])
                sampled_image_level_temp.append(torch.full_like(level_idx, img_idx))
            anchors[img_idx] = anchors_per_level_per_image
            sampled_inds.append(img_sampled_inds)
            sampled_image_levels.append(sampled_image_level_temp)

        # flip back to original format feature map level -> images
        anchors = list(zip(*anchors))

        labels = torch.cat(labels, dim=0)

        # find permutation that brings the concatenated representation in the order
        # that first joins the images for the same level, and then concatenates the
        # levels into the representation obtained by concatenating first the feature maps
        # and then the images
        sampled_image_levels = list(zip(*sampled_image_levels))
        sampled_image_levels = cat([cat(l, dim=0) for l in sampled_image_levels], dim=0)
        permute_inds = cat([nonzero(sampled_image_levels == img_idx)[0]
                for img_idx in range(num_images)], dim=0)

        self._permute_inds = permute_inds

        return anchors

    def prepare_targets(self, anchors, targets):
        """
        This reimplents parts of the functionality of TargetPreparator.__call__
        The reason being that we don't need bbox regression targets for
        masks, so I decided to specialize it here instead of modifying
        TargetPreparator. It might be worth considering modifying this once
        I implement keypoints
        """
        # flip anchor representation to be first images, and then feature maps
        anchors = list(zip(*anchors))
        anchors = [cat_bbox(anchor) for anchor in anchors]

        target_preparator = self.target_preparator
        # TODO assert / resize anchors to have the same .size as targets?
        matched_targets = target_preparator.match_targets_to_anchors(anchors, targets)
        labels = []
        keypoints = []
        for matched_targets_per_image, anchors_per_image in zip(
                matched_targets, anchors):
            labels_per_image, keypoints_per_image = target_preparator.prepare_labels(
                    matched_targets_per_image, anchors_per_image)
            labels.append(labels_per_image)
            keypoints.append(keypoints_per_image)
        return labels, keypoints

    def get_permutation_inds(self, anchors):
        # anchors is in features - images order get the permutation to make it in
        # image - features order
        num_images = len(anchors[0])
        # flatten anchors into a single list
        flattened = [f for l in anchors for f in l]
        sizes = [i.bbox.shape[0] for i in flattened]
        # device = torch.device('cpu')
        device = flattened[0].bbox.device
        # strategy: start from the identity permutation which has the final size
        # split it according to the sizes for each feature map / image, group the
        # indices according to a list of features of list of images, invert the
        # representation to be images -> features, and then concatenate it all
        inds = torch.arange(sum(sizes), device=device)
        # can't use torch.split because of a bug with 0 in sizes
        inds = split_with_sizes(inds, sizes)
        # grouped correspond to the linear indices split in
        # features first, and then images
        grouped = [inds[i:i+num_images] for i in range(0, len(inds), num_images)]
        # convert to images first, then features by flipping the representation
        flip = list(zip(*grouped))
        # flatten the list of lists into a single list of tensors
        flip = [f for l in flip for f in l]
        return torch.cat(flip, dim=0)

    def __call__(self, anchors, keypoint_logits):
        # labels, keypoint_targets = self.prepare_targets(anchors, targets)

        # labels = self._labels
        # keypoints = self._keypoints
        # sampled_pos_inds = torch.cat(self._sampled_pos_inds, dim=0)
        # permute_inds = self._permute_inds

        permutation_inds = self.get_permutation_inds(anchors)
        # from IPython import embed; embed()
        # TODO is this required???
        # mask_logits = cat(mask_logits, dim=0)
        keypoint_logits = keypoint_logits[permutation_inds]
        

        anchors = list(zip(*anchors))
        # anchors = [cat_bbox(anchor) for anchor in anchors]
        keypoints = []
        for anchors_per_image in anchors:
            kp = [anchor.get_field('keypoints').keypoints for anchor in anchors_per_image]
            kp = cat(kp, 0)
            keypoints.append(kp)
        # keypoints = [anchor.get_field('keypoints') for anchor in anchors]
        new_anchors = []
        for anchor in anchors:
            new_anchors.append([a.copy_with_fields([]) for a in anchor])
        # TODO make cat_bbox work with Keypoint
        anchors = [cat_bbox(anchor) for anchor in new_anchors]
        heatmaps = []
        valid = []
        for kp, anchor in zip(keypoints, anchors):
            heatmaps_per_image, valid_per_image = self.target_preparator.project(kp, anchor)
            heatmaps.append(heatmaps_per_image.view(-1))
            valid.append(valid_per_image.view(-1))
        keypoint_targets = cat(heatmaps, dim=0)
        valid = cat(valid, dim=0).to(dtype=torch.uint8)# .bool()
        valid = torch.nonzero(valid).squeeze(1)
        # labels = cat(labels, dim=0)
        # keypoint_targets = cat(targets, dim=0)

        MIN_KEYPOINT_COUNT_FOR_VALID_MINIBATCH = 20
        num_valid = valid.sum().item()
        # torch.mean (in binary_cross_entropy_with_logits) does'nt
        # accept empty tensors, so handle it sepaartely
        if keypoint_targets.numel() == 0 or valid.numel() == 0 or num_valid <= MIN_KEYPOINT_COUNT_FOR_VALID_MINIBATCH:
            return keypoint_logits.sum() * 0

        N, K, H, W = keypoint_logits.shape
        keypoint_logits = keypoint_logits.view(N * K, H * W)

        keypoint_loss = F.cross_entropy(
                keypoint_logits[valid], keypoint_targets[valid])
        return keypoint_loss


def make_roi_keypoint_loss_evaluator(cfg):
    target_preparator = KeypointTargetPreparator(matcher, resolution)
    loss_evaluator = KeypointRCNNLossComputation(target_preparator, fg_bg_sampler)
