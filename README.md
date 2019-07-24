Rotated Mask R-CNN
-----------------
By [Shijie Looi](https://github.com/mrlooi). 

(Paper to be published soon)

This project is based on [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark).

Introduction
-----------------
[Rotated Mask R-CNN]() extends Faster R-CNN, Mask R-CNN, or even RPN only to work with rotated bounding boxes.

This work also builds on the Mask Scoring R-CNN (or 'MS R-CNN') paper ([maskscoring_rcnn](https://github.com/zjhuang22/maskscoring_rcnn)) by learning the quality of the predicted instance masks.

The repo master branch is fully merged upstream with the latest master branch of [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) (as of 24/07/2019)

Additional Features
-----------------
- Soft NMS (Implemented for both bounding box and rotated detections. [Original repo](https://github.com/bharatsingh430/soft-nms))
- Mask IoU head (From [maskscoring_rcnn](https://github.com/zjhuang22/maskscoring_rcnn)) 

Install
-----------------
  Check [INSTALL.md](INSTALL.md) for installation instructions.


Prepare Data
----------------
```
  mkdir -p datasets/coco
  ln -s /path_to_coco_dataset/annotations datasets/coco/annotations
  ln -s /path_to_coco_dataset/train2014 datasets/coco/train2014
  ln -s /path_to_coco_dataset/test2014 datasets/coco/test2014
  ln -s /path_to_coco_dataset/val2014 datasets/coco/val2014
```

Configs
---------------
All example configs related to rotated maskrcnn are in **configs/rotated** folder

Pretrained Models
---------------
Pre-trained models on MSCOCO can be found here [link: TODO]()


Training
----------------
Single GPU Training
```
  python tools/train_net.py --config-file "configs/rotated/e2e_ms_rcnn_R_50_FPN_1x.yaml" SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 SOLVER.MAX_ITER 720000 SOLVER.STEPS "(480000, 640000)" TEST.IMS_PER_BATCH 1
```
Multi-GPU Training
```
  export NGPUS=8
  python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "configs/rotated/e2e_ms_rcnn_R_50_FPN_1x.yaml" 
```

For more details, see README.md in https://github.com/facebookresearch/maskrcnn-benchmark

Testing
----------------
see README.md in https://github.com/facebookresearch/maskrcnn-benchmark

Inference
----------------
```
  python my_tools/infer_demo.py
```
Be sure to change the input values e.g. config_file, model_file, image_dir

Results
------------
Trained on coco/train2014, evaluated on coco/val2014  

| Backbone  | Method | mAP(mask) |
|----------|--------|-----------|
| ResNet-50 FPN | Mask R-CNN | 34.2 |
| ResNet-50 FPN | MS R-CNN | 35.6 |
| ResNet-50 FPN | Rotated Mask R-CNN | 34.2 |
| ResNet-50 FPN | Rotated MS R-CNN | 35.6 |


Examples
-------------

Acknowledgment
-------------
The work was done at [Dorabot Inc](https://www.dorabot.com/).

Citations
---------------
If you find Rotated Mask R-CNN useful in your research, please consider citing:
```   
```

License
---------------
rotated_maskrcnn is released under the MIT license. See [LICENSE](LICENSE) for additional details.

Thanks to the Third Party Libs
---------------  
[maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)   
[Pytorch](https://github.com/pytorch/pytorch)   
