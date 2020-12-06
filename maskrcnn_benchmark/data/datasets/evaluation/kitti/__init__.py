
from .kitti_eval import do_kitti_evaluation as do_orig_kitti_evaluation
from maskrcnn_benchmark.data.datasets import AbstractDataset, KittiDataset

def kitti_evaluation(
    dataset,
    predictions,
    output_folder,
    box_only,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
):
    if isinstance(dataset, KITTIDataset):
        return do_orig_kitti_evaluation(
            dataset=dataset,
            predictions=predictions,
            box_only=box_only,
            output_folder=output_folder,
            iou_types=iou_types,
            expected_results=expected_results,
            expected_results_sigma_tol=expected_results_sigma_tol,
        )
    elif isinstance(dataset, AbstractDataset):
        return do_wrapped_kitti_evaluation(
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
                "Ground truth dataset is not a KITTIDataset, "
                "nor it is derived from AbstractDataset: type(dataset)="
                "%s" % type(dataset)
            )
        )
