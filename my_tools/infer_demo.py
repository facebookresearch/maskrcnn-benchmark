import numpy as np
import cv2

from maskrcnn_benchmark.modeling.rrpn.anchor_generator import draw_anchors
from maskrcnn_benchmark.utils import cv2_util

from predictor import Predictor


def get_random_color():
    return (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))


if __name__ == '__main__':
    import glob

    confidence_threshold = 0.7

    # CLASSES = ["__background__", "text"]

    config_file = "configs/mscoco/mscoco_miou_4x.yaml"
    model_file = "checkpoints/mscoco/mscoco_miou/model_final.pth"
    image_dir = "/data/MSCOCO/val2014"
    image_files = glob.glob("%s/*.jpg"%(image_dir))

    prediction_model = Predictor(config_file, min_score=confidence_threshold, device="cuda")
    prediction_model.load_weights(model_file)

    for image_file in image_files: #glob.glob("%s/*%s"%(image_dir, image_ext)):
        img = cv2.imread(image_file)

        if img is None:
            print("Could not find %s"%(image_file))
            continue

        img_copy = img.copy()

        data = prediction_model.run_on_opencv_image(img)
        if not data:
            print("No predictions for image")
            continue

        scores = data["scores"]
        bboxes = data["bboxes"]
        has_labels = "labels" in data
        has_rrects = "rrects" in data
        has_masks = "masks" in data

        bboxes = np.round(bboxes).astype(np.int32)

        for ix, (bbox, score) in enumerate(zip(bboxes, scores)):

            if has_labels:
                label = data["labels"][ix]

            if has_rrects:
                rr = data["rrects"][ix]
                img_copy = draw_anchors(img_copy, [rr], [[0,0,255]])
            else:
                img_copy = cv2.rectangle(img_copy, tuple(bbox[:2]), tuple(bbox[2:]), (0, 0, 255), 2)

            if has_masks:
                mask = data["masks"][ix]

                contours, hierarchy = cv2_util.findContours(
                    mask.astype(np.uint8) * 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                color = get_random_color()
                img_copy = cv2.drawContours(img_copy, contours, -1, color, 3)

        # from maskrcnn_benchmark.modeling.rotate_ops import merge_rrects_by_iou
        # if has_masks and has_rrects:
        #     img_copy2 = img.copy()
        #
        #     match_inds = merge_rrects_by_iou(data["rrects"], iou_thresh=0.5)
        #
        #     masks = data["masks"]
        #     for idx, inds in match_inds.items():
        #         if len(inds) == 0:
        #             continue
        #         mask = masks[inds[0]]
        #         for ix in inds[1:]:
        #             mask = np.logical_or(mask, masks[ix])
        #
        #         _, contours, hierarchy = cv2.findContours(
        #             mask.astype(np.uint8) * 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        #         )
        #         color = get_random_color()
        #         img_copy2 = cv2.drawContours(img_copy2, contours, -1, color, 3)
        #
        #     cv2.imshow("pred_merged", img_copy2)

        cv2.imshow("img", img)
        cv2.imshow("pred", img_copy)
        cv2.waitKey(0)

