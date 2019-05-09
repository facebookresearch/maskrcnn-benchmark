## Webcam and Jupyter notebook demo

This folder contains a simple webcam demo that illustrates how you can use `maskrcnn_benchmark` for inference.


### With your preferred environment

You can start it by running it from this folder, using one of the following commands:
```bash
# by default, it runs on the GPU
# for best results, use min-image-size 800
python webcam.py --min-image-size 800
# can also run it on the CPU
python webcam.py --min-image-size 300 MODEL.DEVICE cpu
# or change the model that you want to use
python webcam.py --config-file ../configs/caffe2/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml --min-image-size 300 MODEL.DEVICE cpu
# in order to see the probability heatmaps, pass --show-mask-heatmaps
python webcam.py --min-image-size 300 --show-mask-heatmaps MODEL.DEVICE cpu
```

### With Docker

Build the image with the tag `maskrcnn-benchmark` (check [INSTALL.md](../INSTALL.md) for instructions)

Adjust permissions of the X server host (be careful with this step, refer to 
[here](http://wiki.ros.org/docker/Tutorials/GUI) for alternatives)

```bash
xhost +
``` 

Then run a container with the demo:
 
```
docker run --rm -it \
    -e DISPLAY=${DISPLAY} \
    --privileged \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --device=/dev/video0:/dev/video0 \
    --ipc=host maskrcnn-benchmark \
    python demo/webcam.py --min-image-size 300 \
    --config-file configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml
```

**DISCLAIMER:** *This was tested for an Ubuntu 16.04 machine, 
the volume mapping may vary depending on your platform*

## Tracking multiple objects across video frames demo

This folder contains a simple demo that illustrates how you can use `maskrcnn_benchmark` for tracking by detections.
Object tracker uses multiple detections to identify a specific object over time.
There are several algorithms that do it and I decided to use Deep SORT, which is very easy to use and pretty fast. 
[Deep SORT (Simple Online and Realtime Tracking with a Deep Association Metric)](https://arxiv.org/abs/1703.07402) is a 
2017 paper which proposes using a Kalman filter to predict the track of previously identified 
objects, and match them with new detections.

#### Dependencies
```
conda activate maskrcnn_benchmark
conda install scikit-learn
```
Make sure you download the Deep Sort version from my [Github repo](https://github.com/umbertogriffo/deep_sort) since I 
had to make a few small changes to integrate it in this demo.
```
git clone https://github.com/umbertogriffo/deep_sort
```
copy the package **deep_sort/deep_sort** to **maskrcnn-benchmark/demo**

#### Usage

You can start it by running it from this folder, using one of the following commands:

```bash
# by default, it doesn't enable the tracker
# for best results, use min-image-size 800
python video_multi_object_tracking.py --video-file "<path_to_video>" --config-file "../configs/caffe2/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml"
--confidence-threshold 0.7 --min-image-size 800 MODEL.DEVICE cuda MODEL.MASK_ON True 
# can also run it on the CPU
python video_multi_object_tracking.py --video-file "<path_to_video>" --config-file "../configs/caffe2/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml"
--confidence-threshold 0.7 --min-image-size 800 MODEL.DEVICE cpu MODEL.MASK_ON True 
# or enable the tracker
python video_multi_object_tracking.py --video-file "<path_to_video>" --config-file "../configs/caffe2/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml"
--confidence-threshold 0.7 --min-image-size 800 MODEL.DEVICE cpu MODEL.MASK_ON True TRACKER.ENABLE True 
# or enable the tracker and save tracked objects's images to relative folders
python video_multi_object_tracking.py --video-file "<path_to_video>" --config-file "../configs/caffe2/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml"
--confidence-threshold 0.7 --min-image-size 800 MODEL.DEVICE cpu MODEL.MASK_ON True TRACKER.ENABLE True TRACKER.EXTRACT_FROM_MASK.ENABLE True
# or enable the tracker and save tracked objects's images to relative folders with transparent background
python video_multi_object_tracking.py --video-file "<path_to_video>" --config-file "../configs/caffe2/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml"
--confidence-threshold 0.7 --min-image-size 800 MODEL.DEVICE cpu MODEL.MASK_ON True TRACKER.ENABLE True TRACKER.EXTRACT_FROM_MASK.ENABLE True TRACKER.EXTRACT_FROM_MASK.TRANSPARENT True 
# or also resize the images to a specific size
python video_multi_object_tracking.py --video-file "<path_to_video>" --config-file "../configs/caffe2/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml"
--confidence-threshold 0.7 --min-image-size 800 MODEL.DEVICE cpu MODEL.MASK_ON True TRACKER.ENABLE True TRACKER.EXTRACT_FROM_MASK.ENABLE True TRACKER.EXTRACT_FROM_MASK.TRANSPARENT True  TRACKER.EXTRACT_FROM_MASK.DSIZE 800
```