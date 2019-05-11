import numpy as np
import cv2

from maskrcnn_benchmark.modeling.rrpn.anchor_generator import draw_anchors, convert_rect_to_pts, get_bounding_box
from maskrcnn_benchmark.modeling.rotate_ops import crop_min_area_rect

RED = [0,0,255]
PURPLE = [255,0,255]
GREEN = [0,255,0]
BLUE = [255,0,0]
WHITE = [255,255,255]

def warp_image(image, rect):
    center = (rect[0], rect[1])
    width = rect[2]
    height = rect[3]
    theta = np.deg2rad(rect[4])

    v_x = (np.cos(theta), np.sin(theta))
    v_y = (-np.sin(theta), np.cos(theta))
    s_x = center[0] - v_x[0] * ((width-1) / 2) - v_y[0] * ((height-1) / 2)
    s_y = center[1] - v_x[1] * ((width-1) / 2) - v_y[1] * ((height-1) / 2)

    mapping = np.array([[v_x[0],v_y[0], s_x],
                        [v_x[1],v_y[1], s_y]])
    return cv2.warpAffine(image, mapping, image.shape[:2][::-1], flags=cv2.WARP_INVERSE_MAP)


if __name__ == '__main__':
    img = np.zeros((300, 400, 3), dtype=np.uint8)

    img[::60, :] = RED
    img[20::60, :] = GREEN
    img[40::60, :] = BLUE
    roi = np.array([150, 10, 100, 50, 30]) # xc yc w h angle

    img = cv2.imread("/data/MSCOCO/val2014/COCO_val2014_000000415360.jpg")
    roi = [302.46932983, 202.19682312, 284.74664307, 514.6846313, -105.76054382]
    roi = np.round(roi).astype(np.int32)

    im_h, im_w = img.shape[:2]
    shape = (im_w, im_h)

    center = (roi[0], roi[1])
    angle = roi[-1]
    theta = np.deg2rad(angle)
    width = roi[2]
    height = roi[3]

    cropped = crop_min_area_rect(img, roi)

    img_draw = draw_anchors(img, [roi], [PURPLE])
    roi_pts = convert_rect_to_pts(roi)
    roi_bbox = get_bounding_box(roi_pts.reshape((4,2)))
    cv2.rectangle(img_draw, tuple(roi_bbox[:2]), tuple(roi_bbox[2:]), RED)

    cv2.imshow("img", img_draw)
    cv2.imshow("cropped", cropped)
    # cv2.imshow("img_warped", warp_image(img_draw, roi))
    # cv2.waitKey(0)

    RESOLUTION = 8
    resized = cv2.resize(cropped, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_NEAREST)

    resized = cv2.resize(resized, (width, height), interpolation=cv2.INTER_NEAREST)

    # DEBUG
    DEBUG = True
    if DEBUG:
        resized = np.array([[7.03866780e-01, 7.79788256e-01, 4.44556586e-03, 1.07861623e-01,
          3.87215056e-02, 1.09867170e-01, 3.58054250e-01, 3.38938624e-01, ],
         [3.49137604e-01, 8.34304929e-01, 5.80788672e-01, 9.80181754e-01,
          6.74314678e-01, 9.16137576e-01, 8.81183624e-01, 7.32796311e-01, ],
         [1.43708484e-02, 9.83685493e-01, 9.87980187e-01, 9.98000801e-01,
          9.98892128e-01, 9.97464776e-01, 9.67755020e-01, 6.08269215e-01, ],
         [2.96071786e-02, 8.49796474e-01, 9.95640039e-01, 9.95277286e-01,
          9.70138729e-01, 7.16274321e-01, 5.49108684e-01, 2.12217327e-02, ],
         [1.39033133e-02, 9.98789966e-01, 9.99995232e-01, 9.99648690e-01,
          9.19952929e-01, 9.94548574e-03, 2.77294521e-03, 9.53607887e-05, ],
         [2.91091233e-01, 9.98359978e-01, 9.98278975e-01, 9.99483943e-01,
          5.27974129e-01, 8.03469308e-03, 3.94798815e-03, 1.56751124e-03, ],
         [8.32832098e-01, 9.97228205e-01, 9.93853927e-01, 9.52263236e-01,
          1.27692282e-01, 6.55004475e-03, 5.24987758e-04, 1.65246934e-06, ],
         [4.97615606e-01, 7.41151929e-01, 7.08023190e-01, 9.54711214e-02,
          4.17516269e-02, 2.12685484e-03, 2.66465941e-04, 4.61285614e-04, ], ])
        resized = cv2.resize(resized, (width, height), interpolation=cv2.INTER_NEAREST)
        resized[resized >= 0.5] = 255
        resized[resized < 0] = 0
        # resized = cv2.cvtColor(resized.astype(np.float32), cv2.COLOR_GRAY2BGR)

    v_x = (np.cos(theta), np.sin(theta))
    v_y = (-np.sin(theta), np.cos(theta))
    # center = (width // 2, height // 2)
    s_x = center[0] - v_x[0] * ((width-1) / 2) - v_y[0] * ((height-1) / 2)
    s_y = center[1] - v_x[1] * ((width-1) / 2) - v_y[1] * ((height-1) / 2)

    mapping = np.array([[v_x[0],v_y[0], s_x],
                        [v_x[1],v_y[1], s_y]])
    M = mapping

    mask = np.zeros((im_h, im_w), dtype=np.uint8)

    # for x in range(width):
    #     for y in range(height):
    #         dx = int(np.round(M[0,0]*x + M[0,1]*y + M[0,2]))
    #         dy = int(np.round(M[1,0]*x + M[1,1]*y + M[1,2]))
    #
    #         if 0 <= dx < im_w and 0 <= dy < im_h:
    #             mask[dy,dx] = resized[y,x]
    #         # cv2.circle(img_draw, (dx,dy), 3, RED)
    #         # cv2.circle(cropped, (x,y), 3, RED)

    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
    x_grid = x_grid.reshape(-1)
    y_grid = y_grid.reshape(-1)
    map_pts_x = x_grid * M[0, 0] + y_grid * M[0, 1] + M[0, 2]
    map_pts_y = x_grid * M[1, 0] + y_grid * M[1, 1] + M[1, 2]
    map_pts_x = np.round(map_pts_x).astype(np.int32)
    map_pts_y = np.round(map_pts_y).astype(np.int32)

    valid_x = np.logical_and(map_pts_x >= 0, map_pts_x < im_w)
    valid_y = np.logical_and(map_pts_y >= 0, map_pts_y < im_h)
    valid = np.logical_and(valid_x, valid_y)
    mask[map_pts_y[valid], map_pts_x[valid]] = resized[y_grid[valid], x_grid[valid]]

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("mask", mask)
    # cv2.imshow("cropped", cropped)
    cv2.waitKey(0)
 

    # bw = int(roi_bbox[2]-roi_bbox[0])
    # bh = int(roi_bbox[3]-roi_bbox[1])
    # v_x = (np.cos(-theta), np.sin(-theta))
    # v_y = (-np.sin(-theta), np.cos(-theta))
    # center = (bw // 2, bh // 2)
    # s_x = center[0] - v_x[0] * ((width-1) / 2) - v_y[0] * ((height-1) / 2)
    # s_y = center[1] - v_x[1] * ((width-1) / 2) - v_y[1] * ((height-1) / 2)
    #
    # mapping = np.array([[v_x[0],v_y[0], s_x],
    #                     [v_x[1],v_y[1], s_y]])
    #
    # matrix = cv2.getRotationMatrix2D( center=center, angle=-angle, scale=1 )
    # resized_t = cv2.warpAffine(resized, matrix, (bw, bh))#, flags=cv2.WARP_INVERSE_MAP)
    #
    # # #
    # # # mask = cv2.warpAffine(src=mask, M=matrix, dsize=shape)
    # # # img_t = cv2.warpAffine(src=img_t, M=matrix, dsize=shape)
    # # #
    # # # iy, ix = np.where(mask==255)
    # # # bbox = [np.min(ix), np.min(iy), np.max(ix), np.max(iy)] # x1 y1 x2 y2
    # # # cv2.rectangle(img_draw, tuple(bbox[:2]), tuple(bbox[2:]), PURPLE)
    # # # img[bbox[1]:bbox[3],bbox[0]:bbox[2]] = img_t[bbox[1]:bbox[3],bbox[0]:bbox[2]]
    # # #
    # cv2.imshow("resized", resized)
    # # cv2.imshow("img", img_draw)
    # cv2.imshow("mask", mask)
    # cv2.imshow("resized_t", resized_t)
    # # cv2.imshow("img_out", img)
    # cv2.waitKey(0)
