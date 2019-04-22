import numpy as np
import torch
import cv2

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader

from maskrcnn_benchmark.modeling.rrpn.inference import make_rpn_postprocessor, REGRESSION_CN
from maskrcnn_benchmark.modeling.rrpn.loss import make_rpn_loss_evaluator

from maskrcnn_benchmark.modeling.rotated_box_coder import BoxCoder

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList

from maskrcnn_benchmark.modeling.rrpn.anchor_generator import \
    make_anchor_generator as make_rrpn_anchor_generator, convert_rect_to_pts2, draw_anchors

from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.rrpn.utils import get_boxlist_rotated_rect_tensor
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import BalancedPositiveNegativeSampler


def get_feature_maps(image, feature_strides, device='cpu'):
    N, C, H, W = image.shape
    feature_maps = [torch.zeros(N,1,H//s,W//s, device=device) for s in feature_strides]
    return feature_maps

def normalize(x, xmin=None, xmax=None):
    xmin = np.min(x) if xmin is None else xmin
    xmax = np.max(x) if xmax is None else xmax
    nx = x - xmin
    nx /= (xmax - xmin)
    return nx

if __name__ == '__main__':

    config_file = "./configs/coco_rpn_only.yaml"
    try:
        cfg.merge_from_file(config_file)
    except KeyError as e:
        print(e)
    cfg.INPUT.PIXEL_MEAN = [0,0,0]
    cfg.freeze()

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=False,
        start_iter=0,
    )

    device = 'cpu'

    anchor_generator = make_rrpn_anchor_generator(cfg)
    num_anchors = anchor_generator.num_anchors_per_location()

    print(num_anchors)

    box_coder = BoxCoder(weights=None) #cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION
    )

    loss_evaluator = make_rpn_loss_evaluator(cfg, box_coder)

    start_iter = 0
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        images = images.to(device)
        targets = [target.to(device) for target in targets]
        feature_maps = get_feature_maps(images.tensors, cfg.MODEL.RPN.ANCHOR_STRIDE)

        anchors = anchor_generator.forward(images, feature_maps)
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        anchors_cnt = [len(a) for a in anchors]

        labels, regression_targets = loss_evaluator.prepare_targets(anchors, targets)

        sampled_pos_inds, sampled_neg_inds = fg_bg_sampler(labels)

        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        total_pos = sampled_pos_inds.numel()
        total_neg = sampled_neg_inds.numel()

        cumu_cnt = 0

        regression_targets = torch.cat(regression_targets, dim=0)#.cpu().numpy()

        img_tensors = images.tensors

        print(total_pos, total_neg)

        # pos_regression_targets = regression_targets[sampled_pos_inds]
        # print(np.rad2deg(pos_regression_targets[:,-1]))

        for ix,cnt in enumerate(anchors_cnt):
            gt = targets[ix]

            gt_bbox = np.round(gt.bbox.cpu().numpy())
            # gt_mask_instance = gt.get_field("masks")
            # gt_polygons = [p.polygons for p in gt_mask_instance.instances.polygons]
            # for gx,gtp in enumerate(gt_polygons):
            #     gtp = [p.view(-1, 2).cpu().numpy() for p in gtp]
            #     gt_polygons[gx] = gtp
            gt_rrects = get_boxlist_rotated_rect_tensor(gt, "masks").cpu().numpy()

            img_t = img_tensors[ix]
            inds = sampled_pos_inds[cumu_cnt < sampled_pos_inds]
            inds = inds[inds < (cumu_cnt+cnt)]

            pos_anchors = anchors[ix][inds - cumu_cnt]
            reg_targets = regression_targets[inds]

            cumu_cnt += cnt

            anchor_rrects = pos_anchors.get_field("rrects")
            rr = anchor_rrects.cpu().numpy()
            # print(rr)

            assert reg_targets.shape == anchor_rrects.shape

            # reg_targets[gt_135, -1] = 0

            # reg_targets[:, -1] = reg_targets_angles
            print(np.rad2deg(reg_targets[:, -1]))
            proposals = box_coder.decode(reg_targets, anchor_rrects).cpu().numpy()

            img = img_t.cpu().numpy()
            img = np.transpose(img, [1,2,0]).copy()
            # img = img[:,:,::-1]  # rgb to bgr
            # img = normalize(img, 0, 1)
            img *= 255
            img = img.astype(np.uint8)

            img2 = img.copy()
            img3 = img.copy()

            for bbox in gt_bbox:
                cv2.rectangle(img, tuple(bbox[:2]), tuple(bbox[2:]), (0,0,255))
            img = draw_anchors(img, gt_rrects)
            img2 = draw_anchors(img2, rr)
            img3 = draw_anchors(img3, proposals)

            cv2.imshow("gt", img)

            cv2.imshow("match", img2)
            cv2.imshow("proposals", img3)

            cv2.waitKey(0)

        # break
