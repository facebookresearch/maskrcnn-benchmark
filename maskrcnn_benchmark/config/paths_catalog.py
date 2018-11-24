# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os
from . import cfg

class DatasetCatalog(object):
    DATA_DIR = "datasets"

    DATASETS = {
        "coco_2014_train": (
            "coco/train2014",
            "coco/annotations/instances_train2014.json",
        ),
        "coco_2014_val": ("coco/val2014", "coco/annotations/instances_val2014.json"),
        "coco_2014_minival": (
            "coco/val2014",
            "coco/annotations/instances_minival2014.json",
        ),
        "coco_2014_valminusminival": (
            "coco/val2014",
            "coco/annotations/instances_valminusminival2014.json",
        ),
        #"voc_2007_trainval": ("voc/VOC2007/JPEGImages", 'trainval'),
        "voc_2007_test": (
            "voc/VOC2007/JPEGImages",
            'test'),
        "voc_2012_train": (
            "voc/VOC2012/JPEGImages",
            'voc/VOC2012/annotations/voc_2012_train.json'),
        # "voc_2012_trainval": (
        #     "voc/VOC2012/JPEGImages", 'trainval'),
        "voc_2012_val": (
            "voc/VOC2012/JPEGImages",
            'voc/VOC2012/annotations/voc_2012_val.json'),
        "voc_2012_test": (
            "voc/VOC2012/JPEGImages",
            'voc/VOC2012/annotations/voc_2012_test.json'),
    }

#     def evaluate_masks(dataset, all_boxes, all_segms, output_dir):
#     """Evaluate instance segmentation."""
#     logger.info('Evaluating segmentations')
#     not_comp = not cfg.TEST.COMPETITION_MODE
#     if _use_json_dataset_evaluator(dataset):
#         coco_eval = json_dataset_evaluator.evaluate_masks(
#             dataset,
#             all_boxes,
#             all_segms,
#             output_dir,
#             use_salt=not_comp,
#             cleanup=not_comp
#         )
#         mask_results = _coco_eval_to_mask_results(coco_eval)
#     elif _use_cityscapes_evaluator(dataset):
#         cs_eval = cs_json_dataset_evaluator.evaluate_masks(
#             dataset,
#             all_boxes,
#             all_segms,
#             output_dir,
#             use_salt=not_comp,
#             cleanup=not_comp
#         )
#         mask_results = _cs_eval_to_mask_results(cs_eval)
#     else:
#         raise NotImplementedError(
#             'No evaluator for dataset: {}'.format(dataset.name)
#         )
#     return OrderedDict([(dataset.name, mask_results)])

#    def _use_json_dataset_evaluator(dataset):
#        """Check if the dataset uses the general json dataset evaluator."""
#        return dataset.name.find('coco_') > -1 or cfg.TEST.FORCE_JSON_DATASET_EVAL
#    if _use_json_dataset_evaluator(dataset):
#    def voc_info(json_dataset):
#       year = json_dataset.name[4:8]
#       image_set = json_dataset.name[9:]
#       devkit_path = get_devkit_dir(json_dataset.name)
#       assert os.path.exists(devkit_path), \
#         'Devkit directory {} not found'.format(devkit_path)
#       anno_path = os.path.join(
#         devkit_path, 'VOC' + year, 'Annotations', '{:s}.xml')
#       image_set_path = os.path.join(
#         devkit_path, 'VOC' + year, 'ImageSets', 'Main', image_set + '.txt')
#       return dict(
#         year=year,
#         image_set=image_set,
#         devkit_path=devkit_path,
#         anno_path=anno_path,
#         image_set_path=image_set_path)

    @staticmethod
    def get(name):
        if "coco" in name or cfg.FORCE_USE_JSON_ANNOTATION:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs[0]),
                ann_file=os.path.join(data_dir, attrs[1]),
            )
            return dict(
                factory="COCODataset",
                args=args,
            )
        elif "voc" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs[0]),
                split=attrs[0][9:],
            )
            return dict(
                factory="PascalVOCDataset",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog(object):
    S3_C2_DETECTRON_URL = "https://s3-us-west-2.amazonaws.com/detectron"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
    }

    C2_DETECTRON_SUFFIX = "output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl"
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "01_33_49.iAX0mXvW",
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "01_36_30.cUF7QR7I",
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "01_38_50.sNxI7sX7",
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "06_31_39.5MIHi1fZ",
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "01_45_57.ZgkA7hPB",
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "01_48_14.DzEQe4wC",
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "02_31_37.KqyEK4tT",
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "06_35_59.RZotkLKI",
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/"):]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name):
        # Detectron C2 models are stored following the structure
        # prefix/<model_id>/2012_2017_baselines/<model_name>.yaml.<signature>/suffix
        # we use as identifiers in the catalog Caffe2Detectron/COCO/<model_id>/<model_name>
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        suffix = ModelCatalog.C2_DETECTRON_SUFFIX
        # remove identification prefix
        name = name[len("Caffe2Detectron/COCO/"):]
        # split in <model_id> and <model_name>
        model_id, model_name = name.split("/")
        # parsing to make it match the url address from the Caffe2 models
        model_name = "{}.yaml".format(model_name)
        signature = ModelCatalog.C2_DETECTRON_MODELS[name]
        unique_name = ".".join([model_name, signature])
        url = "/".join([prefix, model_id, "12_2017_baselines", unique_name, suffix])
        return url
