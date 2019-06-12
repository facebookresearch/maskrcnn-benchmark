import numpy as np
import torch
import torch.nn.functional as F
import cv2
import pycocotools.mask as mask_util

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.modeling.roi_heads.mask_head.loss import compute_proposal_gt_iou
from maskrcnn_benchmark.modeling.rroi_heads.mask_head.loss import compute_rotated_proposal_gt_iou

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    # config_file = "./configs/icdar/general_text_detector.yaml"
    config_file = "./configs/mscoco/dog_skate_miou.yaml"
    try:
        cfg.merge_from_file(config_file)
    except KeyError as e:
        print(e)
    cfg.INPUT.PIXEL_MEAN = [0,0,0]
    cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN = 0.5
    cfg.INPUT.VERTICAL_FLIP_PROB_TRAIN = 0.5
    cfg.INPUT.ROTATE_PROB_TRAIN = 1.0
    cfg.INPUT.ROTATE_DEGREES_TRAIN = (-45, 45)
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.DATALOADER.SIZE_DIVISIBILITY = 0
    cfg.freeze()

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=False,
        start_iter=0,
    )

    device = 'cpu'

    is_rotated = 1 #cfg.MODEL.ROTATED
    if is_rotated:
        from maskrcnn_benchmark.modeling.rrpn.utils import get_boxlist_rotated_rect_tensor
        from maskrcnn_benchmark.modeling.rrpn.anchor_generator import draw_anchors

    start_iter = 0
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        img_tensors = images.tensors

        for id in range(len(targets)):
            t1 = targets[id]
            im_scale = t1.get_field("scale")

            im1 = img_tensors[id]
            im_np = im1.numpy()
            im_np = np.transpose(im_np, [1,2,0])#[:,:,::-1]  # C,H,W, to H,W,C, then RGB to BGR
            im_copy = im_np.copy().astype(np.uint8)
            h,w,_ = im_np.shape
            cv2.imshow("im", im_copy)
            #
            # angle = 45
            # theta = np.deg2rad(angle)
            #
            # center = [w//2, h//2]
            # v_x = (np.cos(theta), np.sin(theta))
            # v_y = (-np.sin(theta), np.cos(theta))
            #
            # s_x = center[0] - v_x[0] * ((w - 1) / 2) - v_y[0] * ((h - 1) / 2)
            # s_y = center[1] - v_x[1] * ((w - 1) / 2) - v_y[1] * ((h - 1) / 2)
            #
            # M = np.array([[v_x[0], v_y[0], 0.5],
            #               [v_x[1], v_y[1], 0.5]])
            # theta = torch.tensor(M, dtype=torch.float32)
            #
            # imtt = im1.unsqueeze(0)
            # grid = F.affine_grid(theta.unsqueeze(0), imtt.size())
            # im1_warped = F.grid_sample(imtt, grid)
            # im1_warped = im1_warped.squeeze().numpy().astype(np.uint8)
            # im1_warped = np.transpose(im1_warped, [1,2,0])  # CHW to HWC
            #
            # cv2.imshow("im_warped", im1_warped)
            # cv2.waitKey(0)

            m_field = t1.get_field("masks")
            labels = t1.get_field('labels')
            bboxes = t1.bbox

            if is_rotated:
                rrects = get_boxlist_rotated_rect_tensor(t1, "masks", "rrects")

            for ix, label in enumerate(labels):
                seg_mask = m_field[ix]
                sm = seg_mask.get_mask_tensor().numpy()
                p = m_field.instances.polygons[ix]
                m = p.convert_to_binarymask()
                # visualize_mask(m.numpy())
                m = m.numpy()  # uint8 format, 0-1
                m *= 255
                m = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)

                if is_rotated:
                    rr = rrects[ix]
                    proposal = rr.numpy().copy()
                    # proposal[:2] += 15
                    m = draw_anchors(m, [rr], [[0, 0, 255]])
                    # m = draw_anchors(m, [proposal], [[255, 0, 0]])

                    # import time
                    # t = time.time()
                    # print(compute_rotated_proposal_gt_iou(sm, proposal))
                    # # print((time.time() - t))
                else:
                    bbox = bboxes[ix]
                    # proposal = bbox + 10
                    cv2.rectangle(m, tuple(bbox[:2]), tuple(bbox[2:]), (0,0,255), 2)
                    # cv2.rectangle(m, tuple(proposal[:2]), tuple(proposal[2:]), (255,0,0), 2)

                    # cropped_mask = seg_mask.crop(proposal)

                    import time
                    t = time.time()

                    # print(compute_proposal_gt_iou(seg_mask, proposal, cropped_mask))
                    # print((time.time() - t))
                cv2.imshow("mask", m)
                cv2.waitKey(0)

            print(m.shape, im_np.shape)