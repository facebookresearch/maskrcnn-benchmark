# COCO style evaluation for custom datasets derived from AbstractDataset
# by botcs@github

import logging
import os
import json

from maskrcnn_benchmark.data.datasets.coco import COCODataset
from .coco_eval import do_coco_evaluation as orig_evaluation
from .abs_to_coco import convert_abstract_to_coco


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
    logger.info("Converting annotations to COCO format...")
    coco_annotation_dict = convert_abstract_to_coco(dataset)

    dataset_name = dataset.__class__.__name__
    coco_annotation_path = os.path.join(output_folder, dataset_name + ".json")
    logger.info("Saving annotations to %s" % coco_annotation_path)
    with open(coco_annotation_path, "w") as f:
        json.dump(coco_annotation_dict, f, indent=2)

    logger.info("Loading annotations as COCODataset")
    coco_dataset = COCODataset(
        ann_file=coco_annotation_path,
        root="",
        remove_images_without_annotations=False,
        transforms=None,  # transformations should be already saved to the json
    )

    return orig_evaluation(
        dataset=coco_dataset,
        predictions=predictions,
        box_only=box_only,
        output_folder=output_folder,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results,
    )
