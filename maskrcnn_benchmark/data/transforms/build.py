# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T


def build_transforms(cfg, is_train=True, normalize=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_prob = cfg.INPUT.FLIP_PROB_TRAIN  # 0.5
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_prob = 0


    transform_ops = [
        T.Resize(min_size, max_size),
        T.RandomHorizontalFlip(flip_prob),
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
