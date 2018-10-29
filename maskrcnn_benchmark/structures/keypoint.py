import torch


# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1

class Keypoints(object):
    def __init__(self, keypoints, size, mode=None):
        # FIXME remove check once we have better integration with device
        # in my version this would consistently return a CPU tensor
        device = keypoints.device if isinstance(keypoints, torch.Tensor) else torch.device('cpu')
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32, device=device)
        num_keypoints = keypoints.shape[0]
        # TODO remove once support or zero in dim is in
        if num_keypoints > 0:
            keypoints = keypoints.view(num_keypoints, -1, 3)
        
        # TODO should I split them?
        # self.visibility = keypoints[..., 2]
        self.keypoints = keypoints# [..., :2]

        self.size = size
        self.mode = mode

    def crop(self, box):
        raise NotImplementedError()

    def resize(self, size, *args, **kwargs):
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        ratio_w, ratio_h = ratios
        resized_data = self.keypoints.clone()
        resized_data[..., 0] *= ratio_w
        resized_data[..., 1] *= ratio_h
        return type(self)(resized_data, size, self.mode)
        

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT,):
            raise NotImplementedError(
                    "Only FLIP_LEFT_RIGHT implemented")

        flip_inds = type(self).FLIP_INDS
        flipped_data = self.keypoints[:, flip_inds]
        width = self.size[0]
        TO_REMOVE = 1
        # Flip x coordinates
        flipped_data[..., 0] = width - flipped_data[..., 0] - TO_REMOVE

        # Maintain COCO convention that if visibility == 0, then x, y = 0
        inds = flipped_data[..., 2] == 0
        flipped_data[inds] = 0
        return type(self)(flipped_data, self.size, self.mode)

    def to(self, *args, **kwargs):
        return type(self)(self.keypoints.to(*args, **kwargs), self.size, self.mode)

    def __getitem__(self, item):
        return type(self)(self.keypoints[item], self.size, self.mode)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'num_instances={}, '.format(len(self.keypoints))
        s += 'image_width={}, '.format(self.size[0])
        s += 'image_height={})'.format(self.size[1])
        return s


def _create_flip_indices(names, flip_map):
    full_flip_map = flip_map.copy()
    full_flip_map.update({v: k for k, v in flip_map.items()})
    flipped_names = [i if i not in full_flip_map else full_flip_map[i] for i in names]
    flip_indices = [names.index(i) for i in flipped_names]
    return torch.tensor(flip_indices)


class PersonKeypoints(Keypoints):
    NAMES = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]
    FLIP_MAP = {
        'left_eye': 'right_eye',
        'left_ear': 'right_ear',
        'left_shoulder': 'right_shoulder',
        'left_elbow': 'right_elbow',
        'left_wrist': 'right_wrist',
        'left_hip': 'right_hip',
        'left_knee': 'right_knee',
        'left_ankle': 'right_ankle'
    }


# TODO this doesn't look great
PersonKeypoints.FLIP_INDS = _create_flip_indices(PersonKeypoints.NAMES, PersonKeypoints.FLIP_MAP)



# TODO make this nicer, this is a direct translation from C2 (but removing the inner loop)
def keypoints_to_heat_map(keypoints, rois, heatmap_size):
    if rois.numel() == 0:
        return rois.new().long(), rois.new().long()
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = heatmap_size / (rois[:, 2] - rois[:, 0])
    scale_y = heatmap_size / (rois[:, 3] - rois[:, 1])

    offset_x = offset_x[:, None]
    offset_y = offset_y[:, None]
    scale_x = scale_x[:, None]
    scale_y = scale_y[:, None]

    x = keypoints[..., 0]
    y = keypoints[..., 1]

    x_boundary_inds = x == rois[:, 2][:, None]
    y_boundary_inds = y == rois[:, 3][:, None]

    x = (x - offset_x) * scale_x
    x = x.floor().long()
    y = (y - offset_y) * scale_y
    y = y.floor().long()
    
    x[x_boundary_inds] = heatmap_size - 1
    y[y_boundary_inds] = heatmap_size - 1

    valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
    vis = keypoints[..., 2] > 0
    valid = (valid_loc & vis).long()

    lin_ind = y * heatmap_size + x
    heatmaps = lin_ind * valid

    return heatmaps, valid


class HeatMaps(object):
    """
    Not used
    """
    def __init__(self, heatmaps, boxes, size, mode=None):
        self.heatmaps = heatmaps
        self.boxes = boxes
        self.size = size
        self.mode = mode

    def heatmaps_to_keypoints(self):
        """Extract predicted keypoint locations from heatmaps. Output has shape
        (#rois, 4, #keypoints) with the 4 rows corresponding to (x, y, logit, prob)
        for each keypoint.
        """
        # This function converts a discrete image coordinate in a HEATMAP_SIZE x
        # HEATMAP_SIZE image to a continuous keypoint coordinate. We maintain
        # consistency with keypoints_to_heatmap_labels by using the conversion from
        # Heckbert 1990: c = d + 0.5, where d is a discrete coordinate and c is a
        # continuous coordinate.

        maps = self.heatmaps
        rois = self.boxes.bbox

        offset_x = rois[:, 0]
        offset_y = rois[:, 1]

        widths = rois[:, 2] - rois[:, 0]
        heights = rois[:, 3] - rois[:, 1]
        widths = widths.clamp(min=1)
        heights = heights.clamp(min=1)
        widths_ceil = torch.ceil(widths)
        heights_ceil = torch.ceil(heights)

        # NCHW to NHWC for use with OpenCV
        # maps = torch.permute(maps, [0, 2, 3, 1])
        xy_preds = torch.zeros(
            (len(rois), 4, cfg.KRCNN.NUM_KEYPOINTS), dtype=torch.float32)
        for i in range(len(rois)):
            roi_map_width = widths_ceil[i]
            roi_map_height = heights_ceil[i]
            width_correction = widths[i] / roi_map_width
            height_correction = heights[i] / roi_map_height

            roi_map = torch.nn.functional.upsample(maps[i][None],
                    size=(roi_map_height, roi_map_width), mode='bilinear', align_corners=False)
            # roi_map = Image.fromarray(maps[i])
            # roi_map = roi_map.resize((roi_map_width, roi_map_height),
            #         resample=Image.BICUBIC)
            # roi_map = torch.from_numpy(np.array(roi_map, copy=False))

            # Bring back to CHW
            # roi_map = torch.permute(roi_map, [2, 0, 1])
            roi_map_probs = scores_to_probs(roi_map.copy())
            w = roi_map.shape[2]
            for k in range(cfg.KRCNN.NUM_KEYPOINTS):
                pos = roi_map[k, :, :].argmax()
                x_int = pos % w
                y_int = (pos - x_int) // w
                assert (roi_map_probs[k, y_int, x_int] ==
                        roi_map_probs[k, :, :].max())
                x = (x_int + 0.5) * width_correction
                y = (y_int + 0.5) * height_correction
                xy_preds[i, 0, k] = x + offset_x[i]
                xy_preds[i, 1, k] = y + offset_y[i]
                xy_preds[i, 2, k] = roi_map[k, y_int, x_int]
                xy_preds[i, 3, k] = roi_map_probs[k, y_int, x_int]

        return xy_preds

