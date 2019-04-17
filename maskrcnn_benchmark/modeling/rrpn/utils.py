import torch

from ..utils import cat

from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.modeling.rrpn.anchor_generator import convert_pts_to_rect

REGRESSION_CN = 5  # box_regression channels: 4 for bbox, 5 for rotated box (xc,yc,w,h,theta)

def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer

def get_boxlist_rotated_rect_tensor(boxlist, masks_field="masks", rrects_field="rrects"):
    """
    Converts polygons inside the BoxList structure into rotated rectangles (xc,yc,w,h,angle)
    Returns the rotated rectangles as a torch tensor (N, 5)
    """
    device = boxlist.bbox.device

    if boxlist.has_field(rrects_field):
        rrect_tensor = boxlist.get_field(rrects_field)
    else:
        if not boxlist.has_field(masks_field):
            raise ValueError("Targets need '%s' field or '%s' field " \
                             "to compute rrects" % (rrects_field, masks_field))
        # convert masks/polygons to rrects
        masks = boxlist.get_field(masks_field)
        rrect_tensor = get_segmentation_mask_rotated_rect_tensor(masks)

    if rrect_tensor.device != device:
        rrect_tensor = rrect_tensor.to(device)

    return rrect_tensor

def get_segmentation_mask_rotated_rect_tensor(seg_mask):
    """
    Converts polygons inside the segmentation mask structure into rotated rectangles (xc,yc,w,h,angle)
    Returns the rotated rectangles as a torch tensor (N, 5)
    :param seg_mask:
    :return:
    """
    assert isinstance(seg_mask, SegmentationMask)

    if seg_mask.mode == "mask":
        polygon_list = seg_mask.instances.convert_to_polygon()
    elif seg_mask.mode == "poly":
        polygon_list = seg_mask.instances
    else:
        raise NotImplementedError

    polygons = [p.polygons for p in polygon_list.polygons]  # list of list
    N = len(polygons)
    rrect_tensor = torch.zeros((N, 5), dtype=torch.float32)
    if N == 0:
        return rrect_tensor

    device = polygons[0][0].device
    if device != rrect_tensor.device:
        rrect_tensor = rrect_tensor.to(device)

    for ix,poly in enumerate(polygons):
        pp = torch.cat([p.view(-1, 2) for p in poly])  # merge all subpolygons into one polygon
        rrect = convert_pts_to_rect(pp.cpu().numpy())  # convert the polygon into a rotated rect
        for i in range(5):
            rrect_tensor[ix,i] = rrect[i]

    return rrect_tensor

def concat_box_prediction_layers(box_cls, box_regression):
    box_cls_flattened = []
    box_regression_flattened = []
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for box_cls_per_level, box_regression_per_level in zip(
        box_cls, box_regression
    ):
        N, AxC, H, W = box_cls_per_level.shape
        Ax = box_regression_per_level.shape[1]
        A = Ax // REGRESSION_CN
        C = AxC // A
        box_cls_per_level = permute_and_flatten(
            box_cls_per_level, N, A, C, H, W
        )
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(
            box_regression_per_level, N, A, REGRESSION_CN, H, W
        )
        box_regression_flattened.append(box_regression_per_level)
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).reshape(-1, C)
    box_regression = cat(box_regression_flattened, dim=1).reshape(-1, REGRESSION_CN)
    return box_cls, box_regression
