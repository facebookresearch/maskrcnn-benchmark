
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
    import argparse

    from maskrcnn_benchmark.config import cfg
    # from maskrcnn_benchmark.data.datasets import coco
    from maskrcnn_benchmark.engine.inference import evaluate

    parser = argparse.ArgumentParser(
        description="Evaluate predictions from predictions.pth file, COCO format. The predictions.pth"
                    "file is the output of tools/test_net.py. This is usually stored in "
                    "<cfg.OUTPUT_DIR>/inference/<test_dataset>/predictions.pth"
    )
    parser.add_argument(
        "-cfg",
        "--config_file",
        required=True,
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "-td",
        "--test_dataset",
        help="Name of test dataset, MUST be listed in paths_catalog.py > DatasetCatalog > DATASETS",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-f",
        "--folder",
        help="Folder where predictions.pth is stored. Defaults to <cfg.OUTPUT_DIR>/inference/<test_dataset>",
        default="",
        type=str,
    )

    args = parser.parse_args()

    # inputs
    config_file = args.config_file
    TEST_dataset = args.test_dataset

    # load config
    cfg.merge_from_file(config_file)

    predictions_folder = args.folder
    if len(args.folder) == 0:
        predictions_folder = "%s/inference/%s"%(cfg.OUTPUT_DIR, TEST_dataset)
    predictions_file = osp.join(predictions_folder, "predictions.pth")
    assert osp.exists(predictions_file)

    output_folder = predictions_folder

    # load predictions
    predictions = torch.load(predictions_file)

    # load dataset
    dataset = load_dataset(TEST_dataset)

    include_per_class_results = True
    box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=cfg.TEST.EXPECTED_RESULTS,
        expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
        include_per_class_results=include_per_class_results
    )

    result = evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
