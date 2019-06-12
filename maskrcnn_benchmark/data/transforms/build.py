# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T


def build_transforms(cfg, is_train=True, normalize=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_horizontal_prob = cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN
        flip_vertical_prob = cfg.INPUT.VERTICAL_FLIP_PROB_TRAIN
        brightness = cfg.INPUT.BRIGHTNESS
        contrast = cfg.INPUT.CONTRAST
        saturation = cfg.INPUT.SATURATION
        hue = cfg.INPUT.HUE

        rotate_prob = cfg.INPUT.ROTATE_PROB_TRAIN
        rotate_degrees = cfg.INPUT.ROTATE_DEGREES_TRAIN
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_horizontal_prob = 0.0
        flip_vertical_prob = 0.0
        brightness = 0.0
        contrast = 0.0
        saturation = 0.0
        hue = 0.0

        rotate_prob = 0.0
        rotate_degrees = (0.0, 0.0)

    transform_ops = []
    if brightness != 0 or contrast != 0 or saturation != 0 or hue != 0:
        color_jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )
        transform_ops.append(color_jitter)

    transform_ops += [
        T.RandomRotation(rotate_degrees, rotate_prob),
        T.Resize(min_size, max_size),
        T.RandomHorizontalFlip(flip_horizontal_prob),
        T.RandomVerticalFlip(flip_vertical_prob),
        T.ToTensor()
    ]
    if normalize:
        to_bgr255 = cfg.INPUT.TO_BGR255
        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
        )
        transform_ops.append(normalize_transform)

    transform = T.Compose(transform_ops)

    return transform
