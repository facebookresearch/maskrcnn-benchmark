## Model Zoo and Baselines

### Hardware
- 8 NVIDIA V100 GPUs

### Software
- PyTorch version: 1.0.0a0+dd2c487
- CUDA 9.2
- CUDNN 7.1
- NCCL 2.2.13-1

### End-to-end Faster and Mask R-CNN baselines

All the baselines were trained using the exact same experimental setup as in Detectron.
We initialize the detection models with ImageNet weights from Caffe2, the same as used by Detectron.

The pre-trained models are available in the link in the model id.

backbone | type | lr sched | im / gpu | train mem(GB) | train time (s/iter) | total train time(hr) | inference time(s/im) | box AP | mask AP | model id
-- | -- | -- | -- | -- | -- | -- | -- | -- | -- | --
R-50-C4 | Fast | 1x | 1 | 5.8 | 0.4036 | 20.2 | 0.17130 | 34.8 | - | [6358800](https://download.pytorch.org/models/maskrcnn/e2e_faster_rcnn_R_50_C4_1x.pth)
R-50-FPN | Fast | 1x | 2 | 4.4 | 0.3530 | 8.8 | 0.12580 | 36.8 | - | [6358793](https://download.pytorch.org/models/maskrcnn/e2e_faster_rcnn_R_50_FPN_1x.pth)
R-101-FPN | Fast | 1x | 2 | 7.1 | 0.4591 | 11.5 | 0.143149 | 39.1 | - | [6358804](https://download.pytorch.org/models/maskrcnn/e2e_faster_rcnn_R_101_FPN_1x.pth)
X-101-32x8d-FPN | Fast | 1x | 1 | 7.6 | 0.7007 | 35.0 | 0.209965 | 41.2 | - | [6358717](https://download.pytorch.org/models/maskrcnn/e2e_faster_rcnn_X_101_32x8d_FPN_1x.pth)
R-50-C4 | Mask | 1x | 1 | 5.8 | 0.4520 | 22.6 | 0.17796 + 0.028 | 35.6 | 31.5 | [6358801](https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_R_50_C4_1x.pth)
R-50-FPN | Mask | 1x | 2 | 5.2 | 0.4536 | 11.3 | 0.12966 + 0.034 | 37.8 | 34.2 | [6358792](https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_R_50_FPN_1x.pth)
R-101-FPN | Mask | 1x | 2 | 7.9 | 0.5665 | 14.2 | 0.15384 + 0.034 | 40.1 | 36.1 | [6358805](https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_R_101_FPN_1x.pth)
X-101-32x8d-FPN | Mask | 1x | 1 | 7.8 | 0.7562 | 37.8 | 0.21739 + 0.034 | 42.2 | 37.8 | [6358718](https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_X_101_32x8d_FPN_1x.pth)

For person keypoint detection:

backbone | type | lr sched | im / gpu | train mem(GB) | train time (s/iter) | total train time(hr) | inference time(s/im) | box AP | keypoint AP | model id
-- | -- | -- | -- | -- | -- | -- | -- | -- | -- | --
R-50-FPN | Keypoint | 1x | 2 | 5.7 | 0.3771 | 9.4 | 0.10941 | 53.7 | 64.3 | 9981060

### Light-weight Model baselines

We provided pre-trained models for selected FBNet models. 
* All the models are trained from scratched with BN using the training schedule specified below. 
* Evaluation is performed on a single NVIDIA V100 GPU with `MODEL.RPN.POST_NMS_TOP_N_TEST` set to `200`. 

The following inference time is reported:
  * inference total batch=8: Total inference time including data loading, model inference and pre/post preprocessing using 8 images per batch.
  * inference model batch=8: Model inference time only and using 8 images per batch.
  * inference model batch=1: Model inference time only and using 1 image per batch.
  * inferenee caffe2 batch=1: Model inference time for the model in Caffe2 format using 1 image per batch. The Caffe2 models fused the BN to Conv and purely run on C++/CUDA by using Caffe2 ops for rpn/detection post processing.

The pre-trained models are available in the link in the model id.

backbone | type | resolution | lr sched | im / gpu | train mem(GB) | train time (s/iter) | total train time (hr) | inference total batch=8 (s/im) | inference model batch=8 (s/im) | inference model batch=1 (s/im) | inference caffe2 batch=1 (s/im) | box AP | mask AP | model id
-- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | --
[R-50-C4](configs/e2e_faster_rcnn_R_50_C4_1x.yaml) (reference) | Fast | 800 | 1x | 1 | 5.8 | 0.4036 | 20.2 | 0.0875 | **0.0793** | 0.0831 | **0.0625** | 34.4 | - | f35857197
[fbnet_chamv1a](configs/e2e_faster_rcnn_fbnet_chamv1a_600.yaml) | Fast | 600 | 0.75x | 12 | 13.6 | 0.5444 | 20.5 | 0.0315 | **0.0260** | 0.0376 | **0.0188** | 33.5 | - | [f100940543](https://download.pytorch.org/models/maskrcnn/e2e_faster_rcnn_fbnet_chamv1a_600.pth)
[fbnet_default](configs/e2e_faster_rcnn_fbnet_600.yaml) | Fast | 600 | 0.5x | 16 | 11.1 | 0.4872 | 12.5 | 0.0316 | **0.0250** | 0.0297 | **0.0130** | 28.2 | - | [f101086388](https://download.pytorch.org/models/maskrcnn/e2e_faster_rcnn_fbnet_600.pth)
[R-50-C4](configs/e2e_mask_rcnn_R_50_C4_1x.yaml) (reference) | Mask | 800 | 1x | 1 | 5.8 | 0.452 | 22.6 | 0.0918 | **0.0848** | 0.0844 | - | 35.2 | 31.0 | f35858791
[fbnet_xirb16d](configs/e2e_mask_rcnn_fbnet_xirb16d_dsmask_600.yaml) | Mask | 600 | 0.5x | 16 | 13.4 | 1.1732 | 29 | 0.0386 | **0.0319** | 0.0356 | - | 30.7 | 26.9 | [f101086394](https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_fbnet_xirb16d_dsmask.pth)
[fbnet_default](configs/e2e_mask_rcnn_fbnet_600.yaml) | Mask | 600 | 0.5x | 16 | 13.0 | 0.9036 | 23.0 | 0.0327 | **0.0269** | 0.0385 | - | 29.0 | 26.1 | [f101086385](https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_fbnet_600.pth)

## Comparison with Detectron and mmdetection

In the following section, we compare our implementation with [Detectron](https://github.com/facebookresearch/Detectron)
and [mmdetection](https://github.com/open-mmlab/mmdetection).
The same remarks from [mmdetection](https://github.com/open-mmlab/mmdetection/blob/master/MODEL_ZOO.md#training-speed)
about different hardware applies here.

### Training speed

The numbers here are in seconds / iteration. The lower, the better.

type | Detectron (P100) | mmdetection (V100) | maskrcnn_benchmark (V100)
-- | -- | -- | --
Faster R-CNN R-50 C4 | 0.566 | - | 0.4036
Faster R-CNN R-50 FPN | 0.544 | 0.554 | 0.3530
Faster R-CNN R-101 FPN | 0.647 | - | 0.4591
Faster R-CNN X-101-32x8d FPN | 0.799 | - | 0.7007
Mask R-CNN R-50 C4 | 0.620 | - | 0.4520
Mask R-CNN R-50 FPN | 0.889 | 0.690 | 0.4536
Mask R-CNN R-101 FPN | 1.008 | - | 0.5665
Mask R-CNN X-101-32x8d FPN | 0.961 | - | 0.7562

### Training memory

The lower, the better

type | Detectron (P100) | mmdetection (V100) | maskrcnn_benchmark (V100)
-- | -- | -- | --
Faster R-CNN R-50 C4 | 6.3 | - | 5.8
Faster R-CNN R-50 FPN | 7.2 | 4.9 | 4.4
Faster R-CNN R-101 FPN | 8.9 | - | 7.1
Faster R-CNN X-101-32x8d FPN | 7.0 | - | 7.6
Mask R-CNN R-50 C4 | 6.6 | - | 5.8
Mask R-CNN R-50 FPN | 8.6 | 5.9 | 5.2
Mask R-CNN R-101 FPN | 10.2 | - | 7.9
Mask R-CNN X-101-32x8d FPN | 7.7 | - | 7.8

### Accuracy

The higher, the better

type | Detectron (P100) | mmdetection (V100) | maskrcnn_benchmark (V100)
-- | -- | -- | --
Faster R-CNN R-50 C4 | 34.8 | - | 34.8
Faster R-CNN R-50 FPN | 36.7 | 36.7 | 36.8
Faster R-CNN R-101 FPN | 39.4 | - | 39.1
Faster R-CNN X-101-32x8d FPN | 41.3 | - | 41.2
Mask R-CNN R-50 C4 | 35.8 & 31.4 | - | 35.6 & 31.5
Mask R-CNN R-50 FPN | 37.7 & 33.9 | 37.5 & 34.4 | 37.8 & 34.2
Mask R-CNN R-101 FPN | 40.0 & 35.9 | - | 40.1 & 36.1
Mask R-CNN X-101-32x8d FPN | 42.1 & 37.3 | - | 42.2 & 37.8

