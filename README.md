# Faster R-CNN and Mask R-CNN in PyTorch 1.0

### Installation
```bash
# install pycocotools
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install cityscapesScripts
git clone https://github.com/mcordts/cityscapesScripts.git
cd cityscapesScripts/
python setup.py build_ext install

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

cd maskrcnn-benchmark
python setup.py build develop
```

### Training
```bash
python tools/train_net.py --config-file "configs/pascal_voc/e2e_mask_rcnn_R_50_FPN_1x_cocostyle.yaml" SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.00125 SOLVER.MAX_ITER 402000 SOLVER.STEPS "(144000,)" TEST.IMS_PER_BATCH 1 MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 2000 OUTPUT_DIR /output/tf_dir
```
### Predict
运行demo/Mask_R-CNN_demo.ipynb