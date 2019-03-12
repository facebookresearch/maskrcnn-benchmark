# Faster R-CNN for KITTI and BDD100K

This project is based on facebookresearch's [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark).

## Installation
```
conda create --name py36 python=3.6
conda activate py36
conda install -c anaconda numpy
conda install -c anaconda cython
conda install ipython
pip install tqdm yacs
mkdir ~/github
cd ~/github
export INSTALL_DIR=$PWD
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install
cd ~/github
git clone https://github.com/xanderchf/faster-rcnn-KITTI-BDD100k.git
cd faster-rcnn-KITTI-BDD100k
conda install -c pytorch pytorch torchvision cudatoolkit=9.2
python setup.py build develop
unset INSTALL_DIR
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.12.0-cp36-cp36m-linux_x86_64.whl
pip install tensorboardX
```

## Data Preparation
### KITTI
This repo assumes that you followed NVIDIA DIGIT's instruction to prepare the KITTI dataset.

### BDD100K
Prepare the dataset with the below structure:
```
bdd100k/
  images/
    100k/
      train/
      val/
      test/
  labels/
    bdd100k_labels_images_train.json
    bdd100k_labels_images_val.json
```

Finally, create a symlink for the datasets:
```
ln -s {parent directory of kitti and bdd100k} {root directory of this repo}/datasets
```

## Training

### KITTI
```
python tools/train_net.py --config-file "configs/e2e_faster_rcnn_R_101_FPN_1x.yaml" --use_tensorboard MODEL.ROI_BOX_HEAD.NUM_CLASSES 7 DATASETS.TRAIN '("kitti_train",)' OUTPUT_DIR out_kitti
```

### BDD100K
```
python tools/train_net.py --config-file "configs/e2e_faster_rcnn_R_101_FPN_1x.yaml" --use_tensorboard MODEL.ROI_BOX_HEAD.NUM_CLASSES 11 DATASETS.TRAIN '("bdd100k_train",)' OUTPUT_DIR out_bdd
```

For the BDD100K dataset, you might need to clip the gradient for training stability. For other hyperparameters, refer to [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark).


## Inference

Below commands loads saved weights and save the detections in the [BDD format](https://github.com/ucbdrive/bdd-data/blob/master/doc/format.md) in ``{OUTPUT_DIR}/detections``.

### KITTI
```
python tools/test_net.py --weights out_kitti/*.pth --config-file "configs/e2e_faster_rcnn_R_101_FPN_1x.yaml" DATASETS.TEST '("kitti_val",)' OUTPUT_DIR out_kitti
```

### BDD100K
```
python tools/test_net.py --weights out_bdd/*.pth --config-file "configs/e2e_faster_rcnn_R_101_FPN_1x.yaml" DATASETS.TEST '("bdd100k_val",)' OUTPUT_DIR out_bdd
```

## Evaluate
To [evaluate](https://github.com/ucbdrive/bdd-data/blob/master/bdd_data/evaluate.py) the saved detection, run the below command:
```
python tools/evaluate.py --result out_bdd/detections DATASETS.TEST '("bdd100k_val",)'
```

