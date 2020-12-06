# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
from predictor import KITTIDemo

import time


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=224,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "--show-mask-heatmaps",
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=2,
        help="Number of heatmaps per dimension to show",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    ## prepare object that handles inference plus adds predictions on top of image
    #coco_demo = COCODemo(
    #    cfg,
    #    confidence_threshold=args.confidence_threshold,
    #    show_mask_heatmaps=args.show_mask_heatmaps,
    #    masks_per_dim=args.masks_per_dim,
    #    min_image_size=args.min_image_size,
    #)

    #cam = cv2.VideoCapture(0)
    #while True:
    #    start_time = time.time()
    #    ret_val, img = cam.read()
    #    composite = coco_demo.run_on_opencv_image(img)
    #    print("Time: {:.2f} s / img".format(time.time() - start_time))
    #    cv2.imshow("COCO detections", composite)
    #    if cv2.waitKey(1) == 27:
    #        break  # esc to quit
    #cv2.destroyAllWindows()

    # prepare object that handles inference plus adds predictions on top of image
    kitti_demo = KITTIDemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
    )

    count=0

    import glob
    img_list = glob.glob('/home/fusion/workspace/data/kitti/training/image_2/*.png')
    while count < 1000:
        start_time = time.time()
        img = cv2.imread(img_list[count])
        composite = kitti_demo.run_on_opencv_image(img)
        print("Time: {:.2f} s / img".format(time.time() - start_time))
        cv2.imwrite('./save_dir/save_%06d.jpg' % count, composite)
        count += 1

if __name__ == "__main__":
    main()
