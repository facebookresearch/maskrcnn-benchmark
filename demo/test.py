import cv2
import time

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

config_file = "./configs/caffe2/e2e_faster_rcnn_R_50_FPN_1x_caffe2.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
cfg.merge_from_list(["MODEL.WEIGHT", "logs/faster_rcnn_R_50_FPN_1x.pkl"])

start = time.time()
coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)
end = time.time()
print('detection: model loaded in %2.2f sec'%(end-start))

imgfile = 'test_data/data/kite.jpg'
det = 'test_data/res/'
img = cv2.imread(imgfile)

# compute predictions
predictions = coco_demo.run_on_opencv_image(img)
cv2.imwrite(det+imgfile.split('/')[-1], predictions)
