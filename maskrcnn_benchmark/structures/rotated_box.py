import torch

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1

class RotatedBox(object):
    def __init__(self, rbox, size):
        if not isinstance(rbox, torch.Tensor):
            rbox = torch.as_tensor(rbox, dtype=torch.float32, device=torch.device('cpu'))

        if rbox.ndimension() != 2:
            raise ValueError(
                "rbox should have 2 dimensions, got {}".format(rbox.ndimension())
            )
        if rbox.size(-1) != 5:
            raise ValueError(
                "last dimension of rbox should have a "
                "size of 5, got {}".format(rbox.size(-1))
            )

        self.rbox = rbox
        self.size = size

    def crop(self, box):
        raise NotImplementedError()

    def resize(self, size):
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        ratio_w, ratio_h = ratios
        resized_data = self.rbox.clone()
        resized_data[..., 0] *= ratio_w
        resized_data[..., 1] *= ratio_h
        resized_data[..., 2] *= ratio_w
        resized_data[..., 3] *= ratio_h
        return RotatedBox(resized_data, size)

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        image_width, image_height = self.size
        xc = self.rbox[:, 0]
        yc = self.rbox[:, 1]
        if method == FLIP_LEFT_RIGHT:
            TO_REMOVE = 1
            transposed_xc = image_width - xc - TO_REMOVE
            transposed_yc = yc
        elif method == FLIP_TOP_BOTTOM:
            transposed_xc = xc
            transposed_yc = image_height - yc

        rbox = self.rbox.clone()
        rbox[:, 0] = transposed_xc
        rbox[:, 1] = transposed_yc

        return RotatedBox(rbox, self.size)

    def rotate(self, angle):
        raise NotImplementedError("Rotate not implemented")

    def __getitem__(self, item):
        if self.rbox.numel() == 0:
            raise RuntimeError("Indexing empty RotatedBox")
        if isinstance(item, int):
            selected = self.rbox[item:item+1]
        else:
            selected = self.rbox[item]
        return RotatedBox(selected, self.size)

    def area(self):
        box = self.rbox
        area = box[:, 2] * box[:, 3]

        return area

    def cpu(self):
        return self.rbox.cpu()

    def numpy(self):
        return self.rbox.numpy()

    def __len__(self):
        return self.rbox.shape[0]

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'num_instances={}, '.format(len(self.rbox))
        s += 'image_width={}, '.format(self.size[0])
        s += 'image_height={})'.format(self.size[1])
        return s

