# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import time

import cv2

from demo.predictor import COCODemo
from maskrcnn_benchmark.config import cfg


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Video Demo")
    parser.add_argument(
        "--video-file",
        metavar="FILE",
        help="path to video file",
    )
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
    # cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
    # cfg.merge_from_list(["MODEL.MASK_ON", False])
    cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
    )

    cam = cv2.VideoCapture(args.video_file)

    if cam.isOpened():
        # get vcap property
        width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = cam.get(cv2.CAP_PROP_FPS)
        print("width {0:f}:".format(width))
        print("height {0:f}:".format(height))
        print("fps {0:f}:".format(fps))

    current_frame = 0
    success = True

    while success:

        print("current_frame: ", current_frame)

        # Read next image
        success, img = cam.read()

        if success:
            start_time = time.time()
            composite = coco_demo.run_on_opencv_image(img, current_frame)
            print("Time: {:.2f} s / img".format(time.time() - start_time))
            cv2.imshow("Detections", composite)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
            current_frame += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
