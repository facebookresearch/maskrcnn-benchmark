from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import numpy as np

config_file = "./configs/mrcnn.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
# cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=600,
    confidence_threshold=0.95,
)

import glob
import cv2
image_dir = "/home/vincent/Documents/py/ml/Detectron.pytorch/demo/coco_debug_images"
for image_file in glob.glob(image_dir + "/*.jpg"):
	# load image and then run prediction
	# image_file = "/COCO_val2014_000000010012.jpg"

	img = cv2.imread(image_file)
	if img is None:
		print("Could not find %s"%(image_file))
		continue
	# img = np.expand_dims(img,0)
	predictions = coco_demo.run_on_opencv_image(img)
	print("Showing pred results for %s"%(image_file))
	cv2.imshow("pred", predictions)
	cv2.waitKey(0)
