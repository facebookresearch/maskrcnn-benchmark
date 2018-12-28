import torch
import torch.nn.functional as F
# from torchvision.transforms import functional as F

import numpy as np

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1

TORCH_VERSION = torch.__version__
TORCH_VERSION_MAJOR, TORCH_VERSION_MINOR = TORCH_VERSION.split(".")[:2]
TORCH_VERSION_MAJOR = int(TORCH_VERSION_MAJOR)
TORCH_VERSION_MINOR = int(TORCH_VERSION_MINOR)

# from maskrcnn_benchmark.structures.segmentation_mask import Polygons

def bilinear_upsample(tensor, size):
    sz = (int(size[0]), int(size[1]))
    if TORCH_VERSION_MAJOR == 1:
        return F.interpolate(tensor, sz, mode="bilinear", align_corners=True)
    else:
        return F.upsample(tensor, sz, mode="bilinear", align_corners=True)

def flip(tensor, axis):
    assert isinstance(axis, int)
    if TORCH_VERSION_MAJOR == 1:
        return torch.flip(tensor, (axis,))
    else:
        return torch.Tensor(np.array(np.flip(tensor.numpy(), axis)))
    
def flip_lr(tensor):
    # # flip width channel
    return flip(tensor, 3)

def flip_top_bottom(tensor):
    # # flip height channel
    return flip(tensor, 2)

class ObjectMask(object):
    """
    Contains all object masks in the image
        object_masks: tensor of shape (N,?,W,H)
    """

    def __init__(self, object_masks, size):
        """
        object_masks: tensor of shape (N,?,W,H)
        """
        assert isinstance(object_masks, torch.Tensor) and object_masks.shape[2:][::-1] == size  # size is (W,H)
        # assert object_masks.shape[1] == 3
        
        self.object_masks = object_masks
        self.size = size

    def transpose(self, method):
        # print("TRANSPOSE")
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        if method == FLIP_LEFT_RIGHT:
            flipped = flip_lr(self.object_masks)
        else:
            flipped = flip_top_bottom(self.object_masks)

        return ObjectMask(flipped, size=self.size)

    def resize(self, size, *args, **kwargs):
        # print("RESIZE")
        w,h = size
        scaled = bilinear_upsample(self.object_masks, (h, w))
        return ObjectMask(scaled, size=size)

    def crop(self, box):
        # print("CROP")
        bbox = torch.round(box).int()
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        cropped_centers = self.object_masks[:, :, bbox[1]: bbox[3], bbox[0]: bbox[2]]
        return ObjectMask(cropped_centers, size=(w, h))

    @property
    def data(self):
        return self.object_masks

    def to(self, *args, **kwargs):
        return self

    def __iter__(self):
        return iter(self.__getitem__(i) for i in range(len(self.object_masks)))

    def __getitem__(self, item):
        if isinstance(item, int):
            if item >= len(self.object_masks):
                raise IndexError
            selected_vertex = self.object_masks[item:item + 1]
        elif isinstance(item, slice):
            selected_vertex = self.object_masks[item]
        else:
            # advanced indexing on a single dimension
            if isinstance(item, torch.Tensor) and item.dtype == torch.uint8:
                item = item.nonzero()
                item = item.squeeze(1) if item.numel() > 0 else item
                item = item.tolist()
            selected_vertex = self.object_masks[item]
            # for i in item:
            #     selected_vertex.append(self.vertex_centers[i])
        return ObjectMask(selected_vertex, size=self.size)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_object_masks={}, ".format(len(self.object_masks))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={})".format(self.size[1])
        # s += "mode={})".format(self.mode)
        return s

if __name__ == '__main__':
    pass
