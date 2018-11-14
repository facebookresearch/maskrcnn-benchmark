from .coco import coco_evaluation
from .voc import voc_evaluation


def evaluate(dataset_name, dataset, predictions, output_folder, **kwargs):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset_name: Dataset's name, used to select evaluation method.
        dataset: Dataset object
        predictions(list[BoxList]): each item in the list represents the
            prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
        **kwargs: other args.
    Returns:
        evaluation result
    """
    args = dict(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        **kwargs
    )
    if 'coco' in dataset_name:
        return coco_evaluation(**args)
    elif 'voc' in dataset_name:
        return voc_evaluation(**args)
    else:
        raise NotImplementedError('Unsupported dataset type %s.' % dataset_name)
