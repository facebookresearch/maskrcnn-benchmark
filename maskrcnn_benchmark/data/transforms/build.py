# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T


def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_horizontal_prob = cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN
        flip_vertical_prob = cfg.INPUT.VERTICAL_FLIP_PROB_TRAIN
        brightness = cfg.INPUT.BRIGHTNESS
        contrast = cfg.INPUT.CONTRAST
        saturation = cfg.INPUT.SATURATION
        hue = cfg.INPUT.HUE
        angle = cfg.INPUT.ANGLE_TRAIN
        light_vertical_prob = cfg.INPUT.VERTICAL_LIGHT_PROB_TRAIN
        light_vertical_scale = cfg.INPUT.VERTICAL_LIGHT_SCALE_TRAIN
        light_horizontal_prob = cfg.INPUT.HORIZONTAL_LIGHT_PROB_TRAIN
        light_horizontal_scale = cfg.INPUT.HORIZONTAL_LIGHT_SCALE_TRAIN
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_horizontal_prob = 0.0
        flip_vertical_prob = 0.0
        brightness = 0.0
        contrast = 0.0
        saturation = 0.0
        hue = 0.0
        angle = 0.0
        light_vertical_prob = 0.0
        light_vertical_scale = 0.0
        light_horizontal_prob = 0.0
        light_horizontal_scale = 0.0

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )
    color_jitter = T.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )

    transform = T.Compose(
        [
            T.SmallAngleRotate(angle),
            color_jitter,
            T.HorizontalLinearLight(light_horizontal_prob, light_horizontal_scale),
            T.VerticalLinearLight(light_vertical_prob, light_vertical_scale),
            T.Resize(min_size, max_size),
            T.RandomHorizontalFlip(flip_horizontal_prob),
            T.RandomVerticalFlip(flip_vertical_prob),
            T.ToTensor(),
            normalize_transform,
        ]
    )
    return transform
