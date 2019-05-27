
import torch

def load_dataset(dataset_name):
    from maskrcnn_benchmark.data.build import build_dataset, import_file

    paths_catalog = import_file(
        "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog

    dataset = build_dataset((dataset_name, ), transforms=None, dataset_catalog=DatasetCatalog, is_train=False)[0]
    return dataset

if __name__ == '__main__':
    import os, os.path as osp

    from maskrcnn_benchmark.config import cfg
    from maskrcnn_benchmark.data.datasets import coco
    from maskrcnn_benchmark.engine.inference import evaluate

    # inputs
    config_file = "./configs/dog_skate_4.yaml"
    TEST_dataset = "coco_dog_skate"


    predictions_folder = "./checkpoints/dog_skate_4/inference/%s"%(TEST_dataset)
    predictions_file = osp.join(predictions_folder, "predictions.pth")

    output_folder = predictions_folder

    # load config
    cfg.merge_from_file(config_file)
    # load predictions
    predictions = torch.load(predictions_file)

    # load dataset
    dataset = load_dataset(TEST_dataset)
    # coco_root = "/data/MSCOCO"
    # root = "%s/val2014"%(coco_root)
    # ann_file = "%s/annotations/instances_val2014_dog_skate_samples_4.json"%(coco_root)
    # remove_images_without_annotations = True
    # dataset = coco.COCODataset(ann_file, root, remove_images_without_annotations)

    box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=cfg.TEST.EXPECTED_RESULTS,
        expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
    )

    result = evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
