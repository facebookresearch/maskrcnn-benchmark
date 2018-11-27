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

class VertexMask(object):
    """
    Contains all vertex centers in the image
        vertex_centers: tensor of shape (N,3,W,H), where the second channels represents [x direction, y direction, z distance]
    """

    def __init__(self, vertex_centers, size):
        """
        vertex_centers: tensor of shape (N,3,W,H)
        """
        assert isinstance(vertex_centers, torch.Tensor) and vertex_centers.shape[2:][::-1] == size  # size is (W,H)
        assert vertex_centers.shape[1] == 3
        
        self.vertex_centers = vertex_centers # self._generate_vertex_centers()
        self.size = size

    """TODO"""
    def transpose(self, method):
        # print("TRANSPOSE")
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        if method == FLIP_LEFT_RIGHT:
            flipped = flip_lr(self.vertex_centers) 
        else:
            flipped = flip_top_bottom(self.vertex_centers) 

        return VertexMask(flipped, size=self.size)
    """TODO END"""

    def resize(self, size, *args, **kwargs):
        # print("RESIZE")
        w,h = size
        scaled = bilinear_upsample(self.vertex_centers, (h,w))
        return VertexMask(scaled, size=size)

    def crop(self, box):
        # print("CROP")
        w, h = box[2] - box[0], box[3] - box[1]
        cropped_centers = self.vertex_centers[:, :, box[1] : box[3], box[0] : box[2]]
        return VertexMask(cropped_centers, size=(w, h))

    def to(self, *args, **kwargs):
        return self

    def __iter__(self):
        return iter(self.vertex_centers)

    def __getitem__(self, item):
        if isinstance(item, int):
            selected_vertex = self.vertex_centers[item:item+1]
        elif isinstance(item, slice):
            selected_vertex = self.vertex_centers[item]
        else:
            # advanced indexing on a single dimension
            if isinstance(item, torch.Tensor) and item.dtype == torch.uint8:
                item = item.nonzero()
                item = item.squeeze(1) if item.numel() > 0 else item
                item = item.tolist()
            selected_vertex = self.vertex_centers[item]
            # for i in item:
            #     selected_vertex.append(self.vertex_centers[i])
        return VertexMask(selected_vertex, size=self.size)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_vertex_masks={}, ".format(len(self.vertex_centers))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={})".format(self.size[1])
        # s += "mode={})".format(self.mode)
        return s

if __name__ == '__main__':
    pass
