import numpy as np
import torch
import torch.nn.functional as F
import cv2

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.modeling.rrpn.utils import get_boxlist_rotated_rect_tensor
from maskrcnn_benchmark.modeling.rrpn.anchor_generator import draw_anchors

if __name__ == '__main__':
    config_file = "./configs/dog_skate_4.yaml"
    try:
        cfg.merge_from_file(config_file)
    except KeyError as e:
        print(e)
    cfg.INPUT.PIXEL_MEAN = [0,0,0]
    cfg.INPUT.H_FLIP_PROB_TRAIN = 1.0
    cfg.INPUT.V_FLIP_PROB_TRAIN = 1.0
    cfg.freeze()

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=False,
        start_iter=0,
    )

    device = 'cpu'

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

            rrects = get_boxlist_rotated_rect_tensor(t1, "masks", "rrects")

            for ix, label in enumerate(labels):
                p = m_field.instances.polygons[ix]
                m = p.convert_to_binarymask()
                # visualize_mask(m.numpy())
                m = m.numpy()  # uint8 format, 0-1
                m *= 255
                m = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)

                rr = rrects[ix]
                m = draw_anchors(m, [rr], [[0, 0, 255]])

                print(rr)

                cv2.imshow("mask", m)
                cv2.waitKey(0)
