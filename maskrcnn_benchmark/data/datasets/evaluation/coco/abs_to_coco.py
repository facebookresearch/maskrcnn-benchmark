# COCO style evaluation for custom datasets derived from AbstractDataset
# Warning! area is computed using binary maps, therefore results may differ
# because of the precomputed COCO areas
# by botcs@github

import numpy as np
import torch
import pycocotools.mask as mask_util

from maskrcnn_benchmark.data.datasets.abstract import AbstractDataset
from maskrcnn_benchmark.structures.bounding_box import BoxList

import logging
from datetime import datetime
from tqdm import tqdm


def convert_abstract_to_coco(dataset, num_workers=None, chunksize=100):
    """
    Convert any dataset derived from AbstractDataset to COCO style
    for evaluating with the pycocotools lib

    Conversion imitates required fields of COCO instance segmentation
    ground truth files like: ".../annotations/instances_train2014.json"

    After th conversion is done a dict is returned that follows the same
    format as COCO json files.

    By default .coco_eval_wrapper.py saves it to the hard-drive in json format
    and loads it with the maskrcnn_benchmark's default COCODataset

    Args:
        dataset: any dataset derived from AbstractDataset
        num_workers (optional): number of worker threads to parallelize the
            conversion (default is to use all cores for conversion)
        chunk_size (optional): how many entries one thread processes before
            requesting new task. The larger the less overhead there is.
    """

    logger = logging.getLogger("maskrcnn_benchmark.inference")
    assert isinstance(dataset, AbstractDataset)
    # Official COCO annotations have these fields
    # 'info', 'licenses', 'images', 'type', 'annotations', 'categories'
    coco_dict = {}
    coco_dict["info"] = {
        "description": (
            "This is an automatically generated COCO annotation"
            " file using maskrcnn_benchmark"
        ),
        "date_created": "%s" % datetime.now(),
    }
    coco_dict["type"] = "instances"

    images = []
    annotations = []

    if num_workers is None:
        num_workers = torch.multiprocessing.cpu_count()
    else:
        num_workers = min(num_workers, torch.multiprocessing.cpu_count())

    dataset_name = dataset.__class__.__name__
    num_images = len(dataset)
    logger.info(
        (
            "Parsing each entry in "
            "%s, total=%d. "
            "Using N=%d workers and chunksize=%d"
        )
        % (dataset_name, num_images, num_workers, chunksize)
    )

    with torch.multiprocessing.Pool(num_workers) as pool:
        with tqdm(total=num_images) as progress_bar:
            args = [(dataset, idx) for idx in range(num_images)]
            iterator = pool.imap(process_single_image, args, chunksize=100)
            for img_annots_pair in iterator:
                image, per_img_annotations = img_annots_pair

                images.append(image)
                annotations.extend(per_img_annotations)
                progress_bar.update()

    for ann_id, ann in enumerate(annotations, 1):
        ann["id"] = ann_id

    logger.info("Parsing categories:")
    # CATEGORY DATA
    categories = [
        {"id": category_id, "name": name}
        for category_id, name in dataset.id_to_name.items()
        if name != "__background__"
    ]
    # Logging categories
    for cat in categories:
        logger.info(str(cat))

    coco_dict["images"] = images
    coco_dict["annotations"] = annotations
    coco_dict["categories"] = categories
    return coco_dict


def process_single_image(args):
    dataset, idx = args
    # IMAGE DATA
    img_id = idx + 1
    image = {}
    # Official COCO "images" entries have these fields
    # 'license', 'url', 'file_name', 'height', 'width', 'date_captured', 'id'

    img, target, ret_idx = dataset[idx]
    img_info = dataset.get_img_info(idx)
    assert isinstance(img_info, dict)
    image.update(img_info)
    image["width"], image["height"] = target.size

    if "id" not in image.keys():
        # Start indexing from 1 if "id" field is not present
        image["id"] = img_id
    else:
        img_id = image["id"]

    # ANNOTATION DATA
    per_img_annotations = []
    # Official COCO "annotations" entries have these fields
    # 'segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id'

    #

    assert ret_idx == idx, (ret_idx, idx)
    assert isinstance(target, BoxList)

    bboxes = target.convert("xywh").bbox.tolist()
    segm_available = "masks" in target.fields()
    if segm_available:
        masks = target.get_field("masks").get_mask_tensor()  # [N, H, W]
        if masks.dim() == 2:
            masks = masks.unsqueeze(0)
        rles = masks_to_rles(masks)
        """
        !!!WARNING!!!
        At this point the area value differs from the precomputed
        original COCO area values, because we compute the area
        by counting the nonzero entries of the binary mask
        while COCO areas are computed directly from the polygons

        Example:
        Reference image data
        {'license': 2,
         'url': 'http://farm9.staticflickr.com/8035/8024364858_9c41dc1666_z.jpg',
         'file_name': 'COCO_val2014_000000000139.jpg',
         'height': 426,
         'width': 640,
         'date_captured': '2013-11-21 01:34:01',
         'id': 139}

        Original COCO area values
        [  531.8071, 13244.6572,  5833.1182,  2245.3435,  1833.7841,  1289.3734,
           210.1482,  2913.1104,   435.1450,   217.7192,  2089.9749,   338.6089,
           322.5936,   225.6642,  2171.6189,   178.1851,    90.9873,   189.5601,
           120.2320,  2362.4897]

        Area values using the binary masks
        [  531, 13247,  5846,  2251,  1850,  1292,   212,  2922,   439,   224,
           2060,   342,   324,   226,  2171,   178,    90,   187,   120,  2372]
        """
        areas = (masks != 0).sum([1, 2]).tolist()
    else:
        areas = target.area().tolist()

    cat_ids = target.get_field("labels").long().tolist()
    assert len(bboxes) == len(areas) == len(cat_ids)
    num_instances = len(target)
    for ann_idx in range(num_instances):
        annotation = {}
        if segm_available:
            annotation["segmentation"] = rles[ann_idx]
        annotation["area"] = areas[ann_idx]
        annotation["iscrowd"] = 0
        annotation["image_id"] = img_id
        annotation["bbox"] = bboxes[ann_idx]
        annotation["category_id"] = cat_ids[ann_idx]
        per_img_annotations.append(annotation)

    return image, per_img_annotations


def masks_to_rles(masks_tensor):
    # TODO: parallelize
    rles = []
    for instance_mask in masks_tensor:
        np_mask = np.array(instance_mask[:, :, None], order="F")
        rle = mask_util.encode(np_mask)[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        rles.append(rle)

    return rles
