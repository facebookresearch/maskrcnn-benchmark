# COCO style evaluation for custom datasets derived from AbstractDataset
# by botcs@github

import logging
import tempfile
import os
import torch
from collections import OrderedDict
from tqdm import tqdm
import numpy as np


from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.data.datasets.coco import COCODataset
from maskrcnn_benchmark.data.datasets.abstract import AbstractDataset


import pycocotools.mask as mask_util

def do_coco_evaluation(
    dataset,
    predictions,
    box_only,
    output_folder,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    logger.info("Preparing results for COCO format")

    predictions = predictions[:100]

    results = COCOResults(*iou_types)
    logger.info("Evaluating predictions")
    for iou_type in iou_types:
        with tempfile.NamedTemporaryFile() as f:
            file_path = f.name
            if output_folder:
                file_path = os.path.join(output_folder, iou_type + ".json")
            res = evaluate_predictions_on_coco(
                dataset, predictions, file_path, iou_type
            )
            results.update(res)

    logger.info(results)
    check_expected_results(results, expected_results, expected_results_sigma_tol)
    if output_folder:
        torch.save(results, os.path.join(output_folder, "coco_results.pth"))
    
    return results, coco_results


class COCOWrapper(object):
    """
    Mimics ONLY the basic utilities from pycocotools.coco.COCO class which are
    required for and used by pycocotools.coco.COCOeval

    The implementation focuses to cover the bare minimum to make the script 
    running.

    Variable names may make no sense, aim is to intruduce minimal new terms and
    reuse naming wherever possible.

    Handles both predictions and ground truth
    """
    def __init__(self, dataset, every_prediction=None, iou_type="bbox"):
        # follow COCO notation: gt -> ground truth, dt -> detection
        # COCO API requires data to be held in the memory throughout the eval
        # so fingers crossed that segmentation masks converted to RLE can fit in

        self.dataset = dataset
        self.every_prediction = every_prediction
        self.iou_type = iou_type

        # Wrapper holds either GTs or Predictions
        if every_prediction is None:
            # This COCOWrapper instance will hold only GT annotations
            self.gt = self._buildCOCOAnnotations()
            self.dt = None
        else:
            # This COCOWrapper instance will hold only predictions
            self.gt = None
            self.dt = self._buildCOCOPredictions()


    def getAnnIds(self, *args, **kwargs):
        # AnnIds is not a necessary thing
        return 

    def getCatIds(self, *args, **kwargs):
        return list(self.dataset.classid_to_name)

    def getImgIds(self, *args, **kwargs):
        # ImgIds is not a necessary thing, so just send back a list from [0..N]
        return list(range(len(self.dataset)))




    def _buildCOCOAnnotations(self):
        print("Building COCO GT annotations", flush=True)
        desc = "Parsing COCO GT annotations"
        coco_anns = []

        for image_id in tqdm(range(len(self.dataset)), desc=desc):
            _, anns, _ = self.dataset[image_id]
            if self.iou_type == "segm":
                rles = self.extractRLEs(prediction)

            for inst_idx in range(len(anns)):
                # TODO: find out why BoxList indexing would be a problem.
                # Only ranges can be applied ATM.
                ann = anns[inst_idx:inst_idx+1]
                classid = self.dataset.ccid_to_classid[ann.get_field("labels").item()]
                coco_ann = {
                    "id": inst_idx + 1, # NEVER USE 0 FOR INST ID ***
                    "image_id": image_id,
                    "size": ann.size,
                    "bbox": ann.bbox[0].tolist(),
                    "area": ann.area().item(),
                    "category_id": classid,
                    "iscrowd": 0
                }
                if self.iou_type == "bbox":
                    coco_ann["bbox"] = ann.bbox[0].tolist()
                if self.iou_type == "segm":
                    # May not make much sense, but annToRLE function is a requirement
                    coco_ann["segmentation"] = ann.get_field("masks")
                    coco_ann["segmentation"] = self.annToRLE(coco_ann)
                coco_anns.append(coco_ann)
        # COCO API implementation uses a match matrix of 0s to store matches
        # if the object has ID=0 than that object will be never matched...
        # Thanks for Quizhu Li for pointing out this feature!
        return coco_anns


    

    def _buildCOCOPredictions(self):
        print("Building COCO Predictions", flush=True)
        desc = "Parsing predictions"
        coco_preds = []
        # Iterate over images
        for image_id, prediction in tqdm(enumerate(self.every_prediction), desc=desc, total=len(self.every_prediction)):
            if len(prediction) == 0:
                continue

            img_info = self.dataset.get_img_info(image_id)
            width = img_info["width"]
            height = img_info["height"]
            img_size = width, height

            if prediction.size[0] != width or prediction.size[1] != height:
                prediction = prediction.resize(size=img_size)
            
            if self.iou_type == "segm":
                # Convert all masks on this image from Tensor to RLE
                rles = self.extractRLEs(prediction)

            # Iterate over all instances on this image and extend coco_preds
            for inst_idx in range(len(prediction)):
                pred = prediction[inst_idx:inst_idx+1]
                classid = self.dataset.ccid_to_classid[pred.get_field("labels").item()]
                coco_pred = {
                    "id": inst_idx + 1, # NEVER USE 0 FOR INST ID
                    "image_id": image_id,
                    "size": pred.size,
                    "bbox": pred.bbox[0].tolist(),
                    "area": pred.area().item(),
                    "category_id": classid,
                    "score": pred.get_field("scores").item(), # preds differ here
                    "iscrowd": 0
                }
                if self.iou_type == "bbox":
                    coco_pred["bbox"] = pred.bbox[0].tolist()
                if self.iou_type == "segm":
                    # May not make much sense, but annToRLE function is a requirement
                    coco_pred["segmentation"] = rles[inst_idx]

                coco_preds.append(coco_pred)
        return coco_preds


    def loadAnns(self, *args, **kwargs):
        # This is used by COCO API
        if self.dt is None:
            return self.gt
        else:
            return self.dt
            

    def annToRLE(self, ann):
        segm = ann['segmentation']
        h, w = ann['size']

        if isinstance(segm, dict) and "counts" in segm.keys():
            # already rle
            rle = segm
        elif isinstance(segm, SegmentationMask) and segm.mode == 'poly':
            segm = ann.instances.polygons
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = mask_utils.frPyObjects(segm, h, w)
            rle = mask_utils.merge(rles)
        elif isinstance(segm, SegmentationMask) and segm.mode == 'mask':
            np_mask = np.array(segm.instances.masks[0, :, :, None], order="F")
            rle = mask_utils.encode(np_mask)[0]
        else:
            raise RuntimeError("Unknown segmentation format: %s"%segm)

        return rle


    def extractRLEs(self, prediction):
        masks = prediction.get_field("mask")
        image_width, image_height = prediction.size
        if isinstance(masks, torch.Tensor):
            # Masker is necessary only if masks haven't been already resized.
            if list(masks.shape[-2:]) != [image_height, image_width]:
                masker = Masker(threshold=0.5, padding=1)
                masks = masker(masks.expand(1, -1, -1, -1, -1), prediction)
                masks = masks[0]
            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")
        else:
            raise RuntimeError("Unknown segmentation format: %s"%type(masks))

        return rles


def evaluate_predictions_on_coco(
    gt, dt, json_result_file, iou_type="bbox"
):
    # This works with AbstractDataset and the original COCODataset 
    # is guaranteed as well.

    from pycocotools.cocoeval import COCOeval

    # Add COCODataset wrapper required fields that mimics AbstractDataset
    if isinstance(gt, COCODataset):
        gt.classid_to_name = {
            key: value["name"]
            for key, value in gt.coco.cats.items()
        }
        gt.classid_to_ccid = {
            classid: ccid 
            for ccid, classid in enumerate(gt.classid_to_name.keys(), 1)
        }
        gt.ccid_to_classid = {
            ccid: classid 
            for classid, ccid in gt.classid_to_ccid.items()
        }
        gt.name_to_classid = {
            name: classid
            for classid, name in gt.classid_to_name.items()
        }    
        coco_gt = gt


    elif isinstance(gt, AbstractDataset):
        coco_gt = COCOWrapper(coco_gt, iou_type=iou_type)
    else:
        raise NotImplementedError("Ground truth dataset is not a COCODataset, nor it is derived from AbstractDataset")

    # Predictions are wrapped using the coco_gt's fields which is either
    # AbstractDataset OR a modified COCODataset
    coco_dt = COCOWrapper(coco_gt, dt, iou_type=iou_type)

    # Remove COCODataset wrapper if present
    if isinstance(gt, COCODataset):
        coco_gt = coco_gt.coco

    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


class COCOResults(object):
    METRICS = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "box_proposal": [
            "AR@100",
            "ARs@100",
            "ARm@100",
            "ARl@100",
            "AR@1000",
            "ARs@1000",
            "ARm@1000",
            "ARl@1000",
        ],
        "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
    }

    def __init__(self, *iou_types):
        allowed_types = ("box_proposal", "bbox", "segm", "keypoints")
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in COCOResults.METRICS[iou_type]]
            )
        self.results = results

    def update(self, coco_eval):
        if coco_eval is None:
            return
        from pycocotools.cocoeval import COCOeval

        assert isinstance(coco_eval, COCOeval)
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self.results[iou_type]
        metrics = COCOResults.METRICS[iou_type]
        for idx, metric in enumerate(metrics):
            res[metric] = s[idx]

    def __repr__(self):
        s = "COCOResults("
        for iou_type, result in self.results.items():
            s += "{}={.3f},"
        s = s[:1] + ")"
        return repr(self.results)


def check_expected_results(results, expected_results, sigma_tol):
    if not expected_results:
        return

    logger = logging.getLogger("maskrcnn_benchmark.inference")
    for task, metric, (mean, std) in expected_results:
        actual_val = results.results[task][metric]
        lo = mean - sigma_tol * std
        hi = mean + sigma_tol * std
        ok = (lo < actual_val) and (actual_val < hi)
        msg = (
            "{} > {} sanity check (actual vs. expected): "
            "{:.3f} vs. mean={:.4f}, std={:.4}, range=({:.4f}, {:.4f})"
        ).format(task, metric, actual_val, mean, std, lo, hi)
        if not ok:
            msg = "FAIL: " + msg
            logger.error(msg)
        else:
            msg = "PASS: " + msg
            logger.info(msg)