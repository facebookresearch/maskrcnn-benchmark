import numpy as np
import cv2

from maskrcnn_benchmark.modeling.rrpn.anchor_generator import draw_anchors
from maskrcnn_benchmark.modeling.rotate_ops import crop_min_area_rect

RED = [0,0,255]
PURPLE = [255,0,255]
GREEN = [0,255,0]
BLUE = [255,0,0]
WHITE = [255,255,255]

if __name__ == '__main__':
    im_w = 400
    im_h = 300
    shape = (im_w, im_h)
    img = np.zeros((im_h, im_w, 3), dtype=np.uint8)

    img[::60, :] = RED
    img[20::60, :] = GREEN
    img[40::60, :] = BLUE
    roi = np.array([150, 20, 100, 50, -45]) # xc yc w h angle

    center = (roi[0], roi[1])
    theta = roi[-1]
    width = roi[2]
    height = roi[3]

    cropped = crop_min_area_rect(img, roi)

    img_draw = draw_anchors(img, [roi], [WHITE])

    cv2.imshow("img", img_draw)
    cv2.imshow("cropped", cropped)

    RESOLUTION = 8
    resized = cv2.resize(cropped, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_NEAREST)


    matrix = cv2.getRotationMatrix2D(center=center, angle=theta, scale=1)
    img_t = cv2.warpAffine(src=img, M=matrix, dsize=shape)

    x = int( np.round(center[0] - width/2  ))
    y = int( np.round(center[1] - height/2 ))
    h = int(np.round(height))
    w = int(np.round(width))

    xmin = min(max(x, 0), im_w - 1)
    ymin = min(max(y, 0), im_h - 1)

    xmax = min(max(x+w, 0), im_w)
    ymax = min(max(y+h, 0), im_h)

    xmax = max(xmax, xmin + 1)
    ymax = max(ymax, ymin + 1)

    # resized = cv2.resize(resized, (xmax-xmin, ymax-ymin), interpolation=cv2.INTER_NEAREST)
    resized = cv2.resize(resized, (width, height), interpolation=cv2.INTER_NEAREST)

    # img_t
    img_t[ymin:ymax, xmin:xmax] = resized[:ymax-ymin,:xmax-xmin]

    mask = np.zeros((im_h, im_w), dtype=np.uint8)
    mask[ymin:ymax, xmin:xmax] = 255

    matrix = cv2.getRotationMatrix2D(center=center, angle=-theta, scale=1)

    mask = cv2.warpAffine(src=mask, M=matrix, dsize=shape)
    img_t = cv2.warpAffine(src=img_t, M=matrix, dsize=shape)

    iy, ix = np.where(mask==255)
    bbox = [np.min(ix), np.min(iy), np.max(ix), np.max(iy)] # x1 y1 x2 y2
    cv2.rectangle(img_draw, tuple(bbox[:2]), tuple(bbox[2:]), PURPLE)
    img[bbox[1]:bbox[3],bbox[0]:bbox[2]] = img_t[bbox[1]:bbox[3],bbox[0]:bbox[2]]

    cv2.imshow("resized", resized)
    cv2.imshow("img", img_draw)
    cv2.imshow("img_t", img_t)
    cv2.imshow("mask", mask)
    cv2.imshow("img_out", img)
    cv2.waitKey(0)
