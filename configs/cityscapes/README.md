### Paper 
1 [mask-rcnn](https://arxiv.org/pdf/1703.06870.pdf)  


### dataset
1 [cityscapesScripts](https://github.com/mcordts/cityscapesScripts)  


### Performance (from paper)
|      case    | training data | im/gpu | mask AP[val] | mask AP [test] | mask AP50 [test] |
|--------------|:-------------:|:------:|:------------:|:--------------:|-----------------:|
|   R-50-FPN   | fine          |   8/8  |    31.5      | 26.2           | 49.9             |
|   R-50-FPN   | fine + COCO   |   8/8  |    36.4      | 32.0           | 58.1             |


### Note (from paper)
We apply our Mask R-CNN models with the ResNet-FPN-50 backbone; we found the 101-layer counterpart performs similarly due to the small dataset size. We train with image scale (shorter side) randomly sampled from [800, 1024], which reduces overfitting; inference is on a single scale of 1024 pixels. We use a mini-batch size of 1 image per GPU (so 8 on 8 GPUs) and train the model for 24k iterations, starting from a learning rate of 0.01 and reducing it to 0.001 at 18k iterations. It takes ∼4 hours of training on a single 8-GPU machine under this setting.  


### Implemetation (for finetuning from coco trained model)
Step 1: download trained model on coco dataset from [model zoo](https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_R_50_FPN_1x.pth)  
Step 2: do the model surgery on the trained model as below and use it as `pretrained model` for finetuning:    
```python
def clip_weights_from_pretrain_of_coco_to_cityscapes(f, out_file):
	""""""
	# COCO categories for pretty print
	COCO_CATEGORIES = [
	    "__background__",
	    "person",
	    "bicycle",
	    "car",
	    "motorcycle",
	    "airplane",
	    "bus",
	    "train",
	    "truck",
	    "boat",
	    "traffic light",
	    "fire hydrant",
	    "stop sign",
	    "parking meter",
	    "bench",
	    "bird",
	    "cat",
	    "dog",
	    "horse",
	    "sheep",
	    "cow",
	    "elephant",
	    "bear",
	    "zebra",
	    "giraffe",
	    "backpack",
	    "umbrella",
	    "handbag",
	    "tie",
	    "suitcase",
	    "frisbee",
	    "skis",
	    "snowboard",
	    "sports ball",
	    "kite",
	    "baseball bat",
	    "baseball glove",
	    "skateboard",
	    "surfboard",
	    "tennis racket",
	    "bottle",
	    "wine glass",
	    "cup",
	    "fork",
	    "knife",
	    "spoon",
	    "bowl",
	    "banana",
	    "apple",
	    "sandwich",
	    "orange",
	    "broccoli",
	    "carrot",
	    "hot dog",
	    "pizza",
	    "donut",
	    "cake",
	    "chair",
	    "couch",
	    "potted plant",
	    "bed",
	    "dining table",
	    "toilet",
	    "tv",
	    "laptop",
	    "mouse",
	    "remote",
	    "keyboard",
	    "cell phone",
	    "microwave",
	    "oven",
	    "toaster",
	    "sink",
	    "refrigerator",
	    "book",
	    "clock",
	    "vase",
	    "scissors",
	    "teddy bear",
	    "hair drier",
	    "toothbrush",
	]
	# Cityscapes of fine categories for pretty print
	CITYSCAPES_FINE_CATEGORIES = [
	    "__background__",
	    "person",
	    "rider",
	    "car",
	    "truck",
	    "bus",
	    "train",
	    "motorcycle",
	    "bicycle",
	]
	coco_cats = COCO_CATEGORIES
	cityscapes_cats = CITYSCAPES_FINE_CATEGORIES
	coco_cats_to_inds = dict(zip(coco_cats, range(len(coco_cats))))
	cityscapes_cats_to_inds = dict(
		zip(cityscapes_cats, range(len(cityscapes_cats)))
	)

	checkpoint = torch.load(f)
	m = checkpoint['model']

	weight_names = {
		"cls_score": "module.roi_heads.box.predictor.cls_score.weight", 
		"bbox_pred": "module.roi_heads.box.predictor.bbox_pred.weight", 
		"mask_fcn_logits": "module.roi_heads.mask.predictor.mask_fcn_logits.weight", 
	}
	bias_names = {
		"cls_score": "module.roi_heads.box.predictor.cls_score.bias",
		"bbox_pred": "module.roi_heads.box.predictor.bbox_pred.bias", 
		"mask_fcn_logits": "module.roi_heads.mask.predictor.mask_fcn_logits.bias",
	}
	
	representation_size = m[weight_names["cls_score"]].size(1)
	cls_score = nn.Linear(representation_size, len(cityscapes_cats))
	nn.init.normal_(cls_score.weight, std=0.01)
	nn.init.constant_(cls_score.bias, 0)

	representation_size = m[weight_names["bbox_pred"]].size(1)
	class_agnostic = m[weight_names["bbox_pred"]].size(0) != len(coco_cats) * 4
	num_bbox_reg_classes = 2 if class_agnostic else len(cityscapes_cats)
	bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)
	nn.init.normal_(bbox_pred.weight, std=0.001)
	nn.init.constant_(bbox_pred.bias, 0)

	dim_reduced = m[weight_names["mask_fcn_logits"]].size(1)
	mask_fcn_logits = Conv2d(dim_reduced, len(cityscapes_cats), 1, 1, 0)
	nn.init.constant_(mask_fcn_logits.bias, 0)
	nn.init.kaiming_normal_(
		mask_fcn_logits.weight, mode="fan_out", nonlinearity="relu"
	)
	
	def _copy_weight(src_weight, dst_weight):
		for ix, cat in enumerate(cityscapes_cats):
			if cat not in coco_cats:
				continue
			jx = coco_cats_to_inds[cat]
			dst_weight[ix] = src_weight[jx]
		return dst_weight

	def _copy_bias(src_bias, dst_bias, class_agnostic=False):
		if class_agnostic:
			return dst_bias
		return _copy_weight(src_bias, dst_bias)

	m[weight_names["cls_score"]] = _copy_weight(
		m[weight_names["cls_score"]], cls_score.weight
	)
	m[weight_names["bbox_pred"]] = _copy_weight(
		m[weight_names["bbox_pred"]], bbox_pred.weight
	)
	m[weight_names["mask_fcn_logits"]] = _copy_weight(
		m[weight_names["mask_fcn_logits"]], mask_fcn_logits.weight
	)

	m[bias_names["cls_score"]] = _copy_bias(
		m[bias_names["cls_score"]], cls_score.bias
	)
	m[bias_names["bbox_pred"]] = _copy_bias(
		m[bias_names["bbox_pred"]], bbox_pred.bias, class_agnostic
	)
	m[bias_names["mask_fcn_logits"]] = _copy_bias(
		m[bias_names["mask_fcn_logits"]], mask_fcn_logits.bias
	)

	print("f: {}\nout_file: {}".format(f, out_file))
	torch.save(m, out_file)
```
Step 3: modify the `input&weight&solver` configuration in the `yaml` file, like this:  
```
MODEL:
  WEIGHT: "xxx.pth" # the model u save from above code

INPUT:
  MIN_SIZE_TRAIN: (800, 832, 864, 896, 928, 960, 992, 1024, 1024)
  MAX_SIZE_TRAIN: 2048
  MIN_SIZE_TEST: 1024
  MAX_SIZE_TEST: 2048

SOLVER:
  BASE_LR: 0.01
  IMS_PER_BATCH: 8
  WEIGHT_DECAY: 0.0001
  STEPS: (3000,)
  MAX_ITER: 4000
```
Step 4: train the model.  

