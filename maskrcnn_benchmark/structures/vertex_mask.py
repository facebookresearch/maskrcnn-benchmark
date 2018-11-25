import torch
import torch.nn.functional as F
# from torchvision.transforms import functional as F

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1

TORCH_VERSION = torch.__version__
TORCH_VERSION_MAJOR, TORCH_VERSION_MINOR = TORCH_VERSION.split(".")[:2]


class VertexMask(object):
    """
    This class is unfinished and not meant for use yet
    It is supposed to contain the mask for an object as
    a 2d tensor
    """

    def __init__(self, masks, size):
        self.masks = masks
        self.size = size

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        width, height = self.size
        if method == FLIP_LEFT_RIGHT:
            dim = width
            idx = 2
        elif method == FLIP_TOP_BOTTOM:
            dim = height
            idx = 1

        flip_idx = list(range(dim)[::-1])
        flipped_masks = self.masks.index_select(dim, flip_idx)
        return VertexMask(flipped_masks, self.size, self.mode)

    def crop(self, box):
        w, h = box[2] - box[0], box[3] - box[1]

        cropped_masks = self.masks[:, box[1] : box[3], box[0] : box[2]]
        return VertexMask(cropped_masks, size=(w, h), mode=self.mode)

    def resize(self, size, *args, **kwargs):
    	if TORCH_VERSION_MAJOR == '1':
        	mask_resized = F.interpolate(self.masks, size)
        else:
        	mask_resized = F.upsample(self.masks, size)
        return VertexMask(mask_resized, size) 

    def convert(self, mode=None):
    	return self

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_vertex_masks={}, ".format(len(self.masks))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}".format(self.size[1])
        # s += "mode={})".format(self.mode)
        return s

if __name__ == '__main__':
	pass
