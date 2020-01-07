# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

from PIL import Image
import numpy as np
import cv2
import torch
import torchvision
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target=None):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        if target is None:
            return image
        target = target.resize(image.size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target

class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.vflip(image)
            target = target.transpose(1)
        return image, target

class HorizontalLinearLight(object):
    def __init__(self, prob=0.5, lightsacle=50):
        self.prob = prob
        self.lightsacle = lightsacle

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = np.asarray(image.copy())
            h, w, c = image.shape
            x = np.linspace(-1*self.lightsacle, self.lightsacle, w)
            weight = np.expand_dims(x, axis=1)
            weight = weight.repeat(c, axis=1)
            image = image + weight
            image[image < 0] = 0
            image[image > 255] = 255
            image = Image.fromarray(image.astype(np.uint8))
        return image, target

class VerticalLinearLight(object):
    def __init__(self, prob=0.5, lightsacle=50):
        self.prob = prob
        self.lightsacle = lightsacle

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = np.asarray(image.copy())
            h, w, c = image.shape
            x = np.linspace(-1*self.lightsacle, self.lightsacle, h)
            weight = np.expand_dims(x, axis=1)
            weight = weight.repeat(c, axis=1)
            weight = np.expand_dims(weight, axis=1)
            image = image + weight
            image[image < 0] = 0
            image[image > 255] = 255
            image = Image.fromarray(image.astype(np.uint8))
        return image, target

class SmallAngleRotate(object):
    def __init__(self,angle=10):
        self.angle_range = angle

    def __call__(self, image, target):
        self.angle = random.randint(-1*self.angle_range, self.angle_range)
        image = np.asarray(image.copy())
        h, w, _ = image.shape
        cx = w / 2
        cy = h / 2
        M = cv2.getRotationMatrix2D((cx, cy), self.angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
        target = target.rotate(M)
        image = Image.fromarray(image)
        return image, target

class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, target):
        image = self.color_jitter(image)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image
        return image, target

