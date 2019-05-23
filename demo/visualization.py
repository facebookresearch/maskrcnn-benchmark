import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import cv2
from PIL import Image
import numpy as np

pylab.rcParams['figure.figsize'] = 20, 12

from maskrcnn_benchmark.config import cfg
from predictor_tencent import COCODemo


config_file = "../configs/e2e_mask_rcnn_R_50_FPN_1x_Tencent.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda", "OUTPUT_DIR", "../mask_server/"])

coco_demo = COCODemo(
    cfg,
    min_image_size=600,
    confidence_threshold=0.7,
)

import json
with open("../../../Tencent_segmentation_annotations/instances_val2019.json", "r") as f:
    content = json.load(f)
images = content["images"]
import random
random.shuffle(images)
names = [img["file_name"] for img in images][:300]

for name in names:
    output_name = name.split("/")[-1]
    pil_image = Image.open("/home/jim/Documents/Tencent_segmentation/" + name)
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    # compute predictions
    predictions = coco_demo.run_on_opencv_image(image)
    cv2.imwrite("/home/jim/Documents/mask_server_test_vis/"+output_name, predictions[:,:,])
    print(output_name)
