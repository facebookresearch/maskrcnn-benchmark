from .coco_eval import do_coco_evaluation as do_orig_coco_evaluation
from .coco_eval_wrapper import do_coco_evaluation as do_wrapped_coco_evaluation
from maskrcnn_benchmark.data.datasets import AbstractDataset, COCODataset


def coco_evaluation(
    dataset,
    predictions,
    output_folder,
    box_only,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
):
    if isinstance(dataset, COCODataset):
        return do_orig_coco_evaluation(
            dataset=dataset,
            predictions=predictions,
            box_only=box_only,
            output_folder=output_folder,
            iou_types=iou_types,
            expected_results=expected_results,
            expected_results_sigma_tol=expected_results_sigma_tol,
        )
    elif isinstance(dataset, AbstractDataset):
        return do_wrapped_coco_evaluation(
            dataset=dataset,
            predictions=predictions,
            box_only=box_only,
            output_folder=output_folder,
            iou_types=iou_types,
            expected_results=expected_results,
            expected_results_sigma_tol=expected_results_sigma_tol,
        )
    else:
        raise NotImplementedError(
            (
                "Ground truth dataset is not a COCODataset, "
                "nor it is derived from AbstractDataset: type(dataset)="
                "%s" % type(dataset)
            )
        )
