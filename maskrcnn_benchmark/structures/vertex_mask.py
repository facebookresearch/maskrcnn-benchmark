import torch
import torch.nn.functional as F
# from torchvision.transforms import functional as F

import numpy as np

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1

TORCH_VERSION = torch.__version__
TORCH_VERSION_MAJOR, TORCH_VERSION_MINOR = TORCH_VERSION.split(".")[:2]

# from maskrcnn_benchmark.structures.segmentation_mask import Polygons

class VertexMask(object):
    """
    This class relies on polygons of seg_mask_instance to generate binary masks.
    The binary masks are then mapped onto the vertex centers
    """

    def __init__(self, vertex_centers, size):
        # assert isinstance(polygons, list) and len(polygons) > 0 and isinstance(polygons[0], Polygons)
        # assert isinstance(vertex_centers, np.ndarray) and vertex_centers.shape[:2][::-1] == size  # size is (W,H)
        # self.polygons = self.polygons
        """
        vertex_centers: tensor of shape (N,W,H,3)
        """
        assert isinstance(vertex_centers, torch.Tensor) and vertex_centers.shape[1:3][::-1] == size  # size is (W,H)
        assert vertex_centers.shape[-1] == 3
        
        self.vertex_centers = vertex_centers # self._generate_vertex_centers()
        self.size = size

    """TODO"""
    def transpose(self, method):
        print("TRANPOSE")
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        flipped = []
        # for polygon in self.polygons:
        #     flipped.append(polygon.transpose(method))
        return SegmentationMask(flipped, size=self.size)

    def resize(self, size, *args, **kwargs):
        print("RESIZE")
        scaled = self.vertex_centers # TODO
        # for polygon in self.polygons:
        #     scaled.append(polygon.resize(size, *args, **kwargs))
        return VertexMask(scaled, size=size)
    """TODO END"""

    def crop(self, box):
        print("CROP")
        w, h = box[2] - box[0], box[3] - box[1]
        cropped_centers = self.vertex_centers[:, box[1] : box[3], box[0] : box[2]]
        return VertexMask(cropped_centers, size=(w, h))

    def to(self, *args, **kwargs):
        return self

    def __iter__(self):
        return iter(self.vertex_centers)

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            selected_vertex = self.vertex_centers[item:item+1]
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
        s += "image_height={}".format(self.size[1])
        # s += "mode={})".format(self.mode)
        return s

if __name__ == '__main__':
    pass
