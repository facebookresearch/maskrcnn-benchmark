#!/usr/bin/python
#
# The evaluation script for instance-level semantic labeling.
# We use this script to evaluate your approach on the test set.
# You can use the script to evaluate on the validation set.
#
# Please check the description of the "getPrediction" method below
# and set the required environment variables as needed, such that
# this script can locate your results.
# If the default implementation of the method works, then it's most likely
# that our evaluation server will be able to process your results as well.
#
# To run this script, make sure that your results contain text files
# (one for each test set image) with the content:
#   relPathPrediction1 labelIDPrediction1 confidencePrediction1
#   relPathPrediction2 labelIDPrediction2 confidencePrediction2
#   relPathPrediction3 labelIDPrediction3 confidencePrediction3
#   ...
#
# - The given paths "relPathPrediction" point to images that contain
# binary masks for the described predictions, where any non-zero is
# part of the predicted instance. The paths must not contain spaces,
# must be relative to the root directory and must point to locations
# within the root directory.
# - The label IDs "labelIDPrediction" specify the class of that mask,
# encoded as defined in labels.py. Note that the regular ID is used,
# not the train ID.
# - The field "confidencePrediction" is a float value that assigns a
# confidence score to the mask.
#
# Note that this tool creates a file named "gtInstances.json" during its
# first run. This file helps to speed up computation and should be deleted
# whenever anything changes in the ground truth annotations or anything
# goes wrong.

# python imports
from __future__ import print_function, absolute_import, division
import os, sys
import fnmatch
from copy import deepcopy
import io
from contextlib import redirect_stdout
from tqdm import tqdm


import torch
import logging

import numpy as np
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.layers.misc import interpolate

# Cityscapes imports
from cityscapesscripts.helpers.csHelpers import writeDict2JSON
from cityscapesscripts.helpers.csHelpers import ensurePath
from cityscapesscripts.helpers.csHelpers import colors, getColorEntry


######################
# Parameters
######################


# A dummy class to collect all bunch of data
class CArgs(object):
    def __repr__(self):
        """
        A weird looking pretty print for Evaluation Arguments
        """
        longest_key = max([len(str(k)) for k in self.__dict__.keys()])
        longest_val = max([len(str(v)) for v in self.__dict__.values()])
        s = "\n" + "#" * max(79, (longest_key + longest_val + 3)) + "\n"
        for k, v in self.__dict__.items():
            s += "%{}s : %s\n".format(longest_key) % (k, v)
        s += "#" * max(79, (longest_key + longest_val + 3)) + "\n"
        return s


# And a global object of that class
defaultArgs = CArgs()

# Parameters that should be modified by user
defaultArgs.exportBoxFile = os.path.join("evaluationResults", "boxResult.json")
defaultArgs.exportMaskFile = os.path.join("evaluationResults", "maskResult.json")

# overlaps for evaluation
defaultArgs.overlaps = np.arange(0.5, 1.0, 0.05)
# minimum region size for evaluation [pixels]
defaultArgs.minRegionSizes = np.array([100])
# defaultArgs.minRegionSizes     = np.array( [ 400 ] )

defaultArgs.JSONOutput = True
defaultArgs.quiet = False
defaultArgs.csv = False
defaultArgs.colorized = True
defaultArgs.instLabels = []


def matchGtsWithPreds(dataset, predictions):
    """
    Go through the `dataset` and `predictions` one-by-one, and list all
    instances with any non-zero intersection.

    This function handles matching when only BBoxes are used, and when
    instnace segmentation is available it computes the pixel-wise overlap as
    well

    The implementation is heavily based on the original CityScapes eval script:
    https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py


    Original match structure looks like:
    {"filename1":
        "groundTruth":gtInstances
        "prediction":predInstances
    }
    # Filenames are not necessary, replace them with idx


    <gt/pred>Instances=
    {
        "category_name1":[<gt/pred>Instance1, <gt/pred>Instance2, ...]
        "category_name2":[<gt/pred>Instance3, <gt/pred>Instance4, ...]
    ...
    }

    gtInstance=
    {
        "labelID":int(labelID)
        "instID":int(instID)
        "boxArea":np.count_nonzero(npArray binary mask)
        "intersection": pixel count (ONLY IF the dict is in the inner list of a predInstance["matchedGt"])
        "voidIntersection":REMOVE THIS!!!
        "matchedPred":list(predInstance) which has nonzero intersection
    }

    predInstance=
    {
        "imgName":"path/to/input/img"
        "predID":<a counter's current state>
        "labelID":int(labelID)
        "boxArea":pixel count (ONLY IF the dict is in the inner list of a predInstance["matchedGt"])
        "confidence":float(confidence)
        "intersection":np.count_nonzero( np.logical_and( gtNp == gtInstance["instID"] , boolPredInst) )
        "voidIntersection":REMOVE THIS!!!
        "matchedGt":list(gtInstance) which has nonzero intersection
    }
    """

    assert len(dataset) == len(predictions), f"{len(dataset)} != {len(predictions)}"

    matches = []
    for idx in tqdm(range(len(predictions)), desc="Matching Preds with GT"):
        matches.append(matchGtWithPred(dataset, predictions, idx))

    return matches


def isOverlapping(box1, box2):
    x1min, y1min, x1max, y1max = box1
    x2min, y2min, x2max, y2max = box2
    ret = x1min < x2max and x2min < x1max and y1min < y2max and y2min < y1max
    return ret


def getUnionBox(box1, box2):
    x1min, y1min, x1max, y1max = map(int, box1)
    x2min, y2min, x2max, y2max = map(int, box2)

    xmin = min(x1min, x2min)
    ymin = min(y1min, y2min)
    xmax = max(x1max, x2max)
    ymax = max(y1max, y2max)

    unionBox = xmin, ymin, xmax, ymax
    return unionBox


def getIntersectionBox(box1, box2):
    x1min, y1min, x1max, y1max = map(int, box1)
    x2min, y2min, x2max, y2max = map(int, box2)

    xmin = max(x1min, x2min)
    ymin = max(y1min, y2min)
    xmax = min(x1max, x2max)
    ymax = min(y1max, y2max)

    intersectionBox = xmin, ymin, xmax, ymax
    return intersectionBox


def computeBoxIntersection(gt, pred):
    """
    Compute intersection between GT instance and prediction.
    """
    xmin, ymin, xmax, ymax = getIntersectionBox(gt["box"], pred["box"])
    intersection = (xmax - xmin) * (ymax - ymin)
    return intersection


def computeMaskIntersection(gt, gtMask, pred, predMask):
    """
    Compute intersection between GT instance and prediction.
    Increase efficiency by computing elementwise product between masks
    only inside the tight bounding box of the union of the prediction and
    target masks.
    """
    if gtMask is None or predMask is None:
        return 0

    assert gtMask.shape == predMask.shape
    assert len(gtMask.shape) == len(predMask.shape) == 2

    xmin, ymin, xmax, ymax = getUnionBox(gt["box"], pred["box"])
    gtMask_crop = gtMask[ymin:ymax, xmin:xmax]
    predMask_crop = predMask[ymin:ymax, xmin:xmax]

    # elementwise AND
    intersection = torch.sum(torch.mul(gtMask_crop, predMask_crop)).item()
    return intersection


def matchGtWithPred(dataset, predictions, idx):
    # Collect instances from gt and pred separately per image
    # TODO: not parallel! parallelize this process safely
    perImgGtInstances, gtMasks = prepareGtImage(dataset, idx)
    perImgPredInstances, predMasks = preparePredImage(dataset, predictions, idx)

    # If no masks are provided, the segmentation score will be 0
    for gt, gtMask in zip(perImgGtInstances, gtMasks):
        for pred, predMask in zip(perImgPredInstances, predMasks):
            if not isOverlapping(gt["box"], pred["box"]):
                continue

            boxIntersection = computeBoxIntersection(gt, pred)
            maskIntersection = computeMaskIntersection(gt, gtMask, pred, predMask)

            if boxIntersection > 0:
                # Copy metadata only, and register the matched pairs
                # this step is redundant but informative
                # intersection score would be enough
                gtCopy = gt.copy()
                predCopy = pred.copy()

                # remove linking field (an empty list) to avoid confusion
                gtCopy.pop("matchedPred")
                predCopy.pop("matchedGt")

                gtCopy["boxIntersection"] = boxIntersection
                gtCopy["maskIntersection"] = maskIntersection
                predCopy["boxIntersection"] = boxIntersection
                predCopy["maskIntersection"] = maskIntersection

                gt["matchedPred"].append(predCopy)
                pred["matchedGt"].append(gtCopy)

    # Group by classes
    groupedGtInstances = {labelName: [] for labelName in dataset.CLASSES}
    groupedPredInstances = {labelName: [] for labelName in dataset.CLASSES}

    for gt in perImgGtInstances:
        gtLabelName = dataset.id_to_name[gt["labelID"]]
        groupedGtInstances[gtLabelName].append(gt)

    for pred in perImgPredInstances:
        predLabelName = dataset.id_to_name[pred["labelID"]]
        groupedPredInstances[predLabelName].append(pred)

    match = {"groundTruth": groupedGtInstances, "prediction": groupedPredInstances}

    return match


def prepareGtImage(dataset, idx):
    _, perImageGts, _ = dataset[idx]
    perImageInstances = []
    maskTensor = [None] * len(perImageGts)
    if len(perImageGts) == 0:
        return perImageInstances, maskTensor

    # Resize to original image size
    imgInfo = dataset.get_img_info(idx)
    origSize = imgInfo["width"], imgInfo["height"]
    if perImageGts.size != origSize:
        perImageGts = perImageGts.resize(size=origSize)

    # Compute box areas
    perImageGts = perImageGts.convert("xyxy")
    bbs = perImageGts.bbox.long()
    xmins, ymins, xmaxs, ymaxs = bbs[:, 0], bbs[:, 1], bbs[:, 2], bbs[:, 3]
    boxAreas = ((xmaxs - xmins) * (ymaxs - ymins)).tolist()
    bbs = bbs.tolist()

    # object label for each prediction
    labels = perImageGts.get_field("labels").tolist()
    if "masks" in perImageGts.fields():
        # Get the binary mask for each instance in a contiguous array
        maskTensor = perImageGts.get_field("masks").get_mask_tensor()

        # In case of single mask then add a new axis
        if len(maskTensor.shape) == 2:
            maskTensor = maskTensor[None]

        # unique_values = set(torch.unique(maskTensor).tolist())
        # assert len(unique_values) == 2, "Not binary mask: %s" % unique_values
        # pixelCounts = maskTensor.clamp_(0, 1).sum(dim=[1, 2])
        pixelCounts = []
        for (xmin, ymin, xmax, ymax), instanceMask in zip(bbs, maskTensor):
            pixelCounts.append(instanceMask[ymin:ymax, xmin:xmax].sum().item())

    for instID in range(len(perImageGts)):
        xmin, ymin, xmax, ymax = bbs[instID]
        pixelCount = pixelCounts[instID] if maskTensor[0] is not None else 0
        gtInstance = {
            "labelID": labels[instID],
            "instID": instID,
            "boxArea": boxAreas[instID],
            "pixelCount": pixelCount,
            "box": (xmin, ymin, xmax, ymax),
            "matchedPred": [],
        }
        perImageInstances.append(gtInstance)

    return perImageInstances, maskTensor


def preparePredImage(dataset, predictions, idx):
    perImagePredictions = predictions[idx]

    # A list will hold statistics and meta-data about the image
    perImageInstances = []

    # maskTensor represents binary masks of all predicted instance segmentations
    # if present
    maskTensor = [None] * len(perImagePredictions)

    # No predictions for this image
    if len(perImagePredictions) == 0:
        return perImageInstances, maskTensor

    # Resize to original image size
    imgInfo = dataset.get_img_info(idx)
    origSize = imgInfo["width"], imgInfo["height"]
    if perImagePredictions.size != origSize:
        perImagePredictions = perImagePredictions.resize(size=origSize)

    # Bounding boxes and areas
    perImagePredictions = perImagePredictions.convert("xyxy")
    bbs = perImagePredictions.bbox.long()
    xmins, ymins, xmaxs, ymaxs = bbs[:, 0], bbs[:, 1], bbs[:, 2], bbs[:, 3]
    boxAreas = ((xmaxs - xmins) * (ymaxs - ymins)).tolist()
    bbs = bbs.tolist()

    # object label and "Objectness" score for each prediction
    labels = perImagePredictions.get_field("labels").tolist()
    scores = perImagePredictions.get_field("scores").tolist()

    # Get the mask for each instance in a contiguous array
    if "mask" in perImagePredictions.fields():
        maskTensor = perImagePredictions.get_field("mask")

        # sanity checks
        assert len(perImagePredictions) == len(maskTensor), (
            "number of masks (%d) do not match the number of boxes (%d)"
            % (len(perImagePredictions), len(maskTensor))
        )

        maskTensor = maskTensor.float()
        # We assume that the maskTensors are coming right out of the maskRCNN
        # having values between [0, 1] inclusive
        #
        # assert maskTensor.min() >= 0.0 and maskTensor.max() <= 1.0, [
        #     maskTensor.max(),
        #     maskTensor.min(),
        # ]

        # Project masks to the boxes
        # TODO: Issue #527 - bad Masker interface
        #
        # The predicted masks are in the shape of i.e. [N, 1, 28, 28] where N is
        # the number of instances predicted, and they represent the interior
        # of the bounding boxes.
        #
        # Masker projects these predictions on an empty canvas with the full
        # size of the input image using the predicted bounding boxes
        maskTensor = Masker(threshold=0.5).forward_single_image(
            maskTensor, perImagePredictions
        )[:, 0, :, :]

        pixelCounts = []
        for (xmin, ymin, xmax, ymax), instanceMask in zip(bbs, maskTensor):
            pixelCounts.append(instanceMask[ymin:ymax, xmin:xmax].sum().item())

    for predID in range(len(perImagePredictions)):
        xmin, ymin, xmax, ymax = bbs[predID]
        # if we have instance segmentation prediction then we update pixelCount
        pixelCount = 0
        if maskTensor[0] is not None:
            pixelCount = pixelCounts[predID]
            if pixelCount == 0:
                continue

        predInstance = {
            "imgName": idx,
            "predID": predID,
            "labelID": labels[predID],
            "boxArea": boxAreas[predID],
            "pixelCount": pixelCount,
            "confidence": scores[predID],
            "box": (xmin, ymin, xmax, ymax),
            "matchedGt": [],
        }
        perImageInstances.append(predInstance)

    return perImageInstances, maskTensor


def evaluateBoxMatches(matches, args):
    # In the end, we need two vectors for each class and for each overlap
    # The first vector (y_true) is binary and is 1, where the ground truth says true,
    # and is 0 otherwise.
    # The second vector (y_score) is float [0...1] and represents the confidence of
    # the prediction.
    #
    # We represent the following cases as:
    #                                       | y_true |   y_score
    #   gt instance with matched prediction |    1   | confidence
    #   gt instance w/o  matched prediction |    1   |     0.0
    #          false positive prediction    |    0   | confidence
    #
    # The current implementation makes only sense for an overlap threshold >= 0.5,
    # since only then, a single prediction can either be ignored or matched, but
    # never both. Further, it can never match to two gt instances.
    # For matching, we vary the overlap and do the following steps:
    #   1.) remove all predictions that satisfy the overlap criterion with an ignore region (either void or *group)
    #   2.) remove matches that do not satisfy the overlap
    #   3.) mark non-matched predictions as false positive

    # AP
    overlaps = args.overlaps
    # region size
    minRegionSizes = args.minRegionSizes

    # only keep the first, if distances are not available
    # if not args.distanceAvailable:
    #     minRegionSizes = [ minRegionSizes[0] ]
    #     distThs        = [ distThs       [0] ]
    #     distConfs      = [ distConfs     [0] ]

    # Here we hold the results
    # First dimension is class, second overlap
    ap = np.zeros((len(minRegionSizes), len(args.instLabels), len(overlaps)), np.float)

    for dI, minRegionSize in enumerate(minRegionSizes):
        for (oI, overlapTh) in enumerate(overlaps):
            for (lI, labelName) in enumerate(args.instLabels):
                y_true = np.empty(0)
                y_score = np.empty(0)
                # count hard false negatives
                hardFns = 0
                # found at least one gt and predicted instance?
                haveGt = False
                havePred = False

                for img in matches:
                    predInstances = img["prediction"][labelName]
                    gtInstances = img["groundTruth"][labelName]
                    # filter groups in ground truth
                    gtInstances = [
                        gt for gt in gtInstances if gt["boxArea"] >= minRegionSize
                    ]

                    if gtInstances:
                        haveGt = True
                    if predInstances:
                        havePred = True

                    curTrue = np.ones(len(gtInstances))
                    curScore = np.ones(len(gtInstances)) * (-float("inf"))
                    curMatch = np.zeros(len(gtInstances), dtype=np.bool)

                    # collect matches
                    for (gtI, gt) in enumerate(gtInstances):
                        foundMatch = False
                        for pred in gt["matchedPred"]:
                            overlap = float(pred["boxIntersection"]) / (
                                gt["boxArea"]
                                + pred["boxArea"]
                                - pred["boxIntersection"]
                            )
                            if overlap > overlapTh:
                                # the score
                                confidence = pred["confidence"]

                                # if we already hat a prediction for this groundtruth
                                # the prediction with the lower score is automatically a false positive
                                if curMatch[gtI]:
                                    maxScore = max(curScore[gtI], confidence)
                                    minScore = min(curScore[gtI], confidence)
                                    curScore[gtI] = maxScore
                                    # append false positive
                                    curTrue = np.append(curTrue, 0)
                                    curScore = np.append(curScore, minScore)
                                    curMatch = np.append(curMatch, True)
                                # otherwise set score
                                else:
                                    foundMatch = True
                                    curMatch[gtI] = True
                                    curScore[gtI] = confidence

                        if not foundMatch:
                            hardFns += 1

                    # remove non-matched ground truth instances
                    curTrue = curTrue[curMatch == True]
                    curScore = curScore[curMatch == True]

                    # collect non-matched predictions as false positive
                    for pred in predInstances:
                        foundGt = False
                        for gt in pred["matchedGt"]:
                            overlap = float(gt["boxIntersection"]) / (
                                gt["boxArea"] + pred["boxArea"] - gt["boxIntersection"]
                            )
                            if overlap > overlapTh:
                                foundGt = True
                                break
                        if not foundGt:
                            # collect number of void and *group pixels
                            nbIgnorePixels = 0
                            for gt in pred["matchedGt"]:
                                # small ground truth instances
                                if gt["boxArea"] < minRegionSize:
                                    nbIgnorePixels += gt["boxIntersection"]
                            if pred["boxArea"] > 0:
                                proportionIgnore = (
                                    float(nbIgnorePixels) / pred["boxArea"]
                                )
                            else:
                                proportionIgnore = 0
                            # if not ignored
                            # append false positive
                            if proportionIgnore <= overlapTh:
                                curTrue = np.append(curTrue, 0)
                                confidence = pred["confidence"]
                                curScore = np.append(curScore, confidence)

                    # append to overall results
                    y_true = np.append(y_true, curTrue)
                    y_score = np.append(y_score, curScore)

                # compute the average precision
                if haveGt and havePred:
                    # compute precision recall curve first

                    # sorting and cumsum
                    scoreArgSort = np.argsort(y_score)
                    yScoreSorted = y_score[scoreArgSort]
                    yTrueSorted = y_true[scoreArgSort]
                    yTrueSortedCumsum = np.cumsum(yTrueSorted)

                    # unique thresholds
                    (thresholds, uniqueIndices) = np.unique(
                        yScoreSorted, return_index=True
                    )

                    # since we need to add an artificial point to the precision-recall curve
                    # increase its length by 1
                    nbPrecRecall = len(uniqueIndices) + 1

                    # prepare precision recall
                    nbExamples = len(yScoreSorted)
                    nbTrueExamples = yTrueSortedCumsum[-1]
                    precision = np.zeros(nbPrecRecall)
                    recall = np.zeros(nbPrecRecall)

                    # deal with the first point
                    # only thing we need to do, is to append a zero to the cumsum at the end.
                    # an index of -1 uses that zero then
                    yTrueSortedCumsum = np.append(yTrueSortedCumsum, 0)

                    # deal with remaining
                    for idxRes, idxScores in enumerate(uniqueIndices):
                        cumSum = yTrueSortedCumsum[idxScores - 1]
                        tp = nbTrueExamples - cumSum
                        fp = nbExamples - idxScores - tp
                        fn = cumSum + hardFns
                        p = float(tp) / (tp + fp)
                        r = float(tp) / (tp + fn)
                        precision[idxRes] = p
                        recall[idxRes] = r

                    # first point in curve is artificial
                    precision[-1] = 1.0
                    recall[-1] = 0.0

                    # compute average of precision-recall curve
                    # integration is performed via zero order, or equivalently step-wise integration
                    # first compute the widths of each step:
                    # use a convolution with appropriate kernel, manually deal with the boundaries first
                    recallForConv = np.copy(recall)
                    recallForConv = np.append(recallForConv[0], recallForConv)
                    recallForConv = np.append(recallForConv, 0.0)

                    stepWidths = np.convolve(recallForConv, [-0.5, 0, 0.5], "valid")

                    # integrate is now simply a dot product
                    apCurrent = np.dot(precision, stepWidths)

                elif haveGt:
                    apCurrent = 0.0
                else:
                    apCurrent = float("nan")
                ap[dI, lI, oI] = apCurrent

    return ap


def evaluateMaskMatches(matches, args):
    # In the end, we need two vectors for each class and for each overlap
    # The first vector (y_true) is binary and is 1, where the ground truth says true,
    # and is 0 otherwise.
    # The second vector (y_score) is float [0...1] and represents the confidence of
    # the prediction.
    #
    # We represent the following cases as:
    #                                       | y_true |   y_score
    #   gt instance with matched prediction |    1   | confidence
    #   gt instance w/o  matched prediction |    1   |     0.0
    #          false positive prediction    |    0   | confidence
    #
    # The current implementation makes only sense for an overlap threshold >= 0.5,
    # since only then, a single prediction can either be ignored or matched, but
    # never both. Further, it can never match to two gt instances.
    # For matching, we vary the overlap and do the following steps:
    #   1.) remove all predictions that satisfy the overlap criterion with an ignore region (either void or *group)
    #   2.) remove matches that do not satisfy the overlap
    #   3.) mark non-matched predictions as false positive

    # AP
    overlaps = args.overlaps
    # region size
    minRegionSizes = args.minRegionSizes

    # only keep the first, if distances are not available
    # if not args.distanceAvailable:
    #     minRegionSizes = [ minRegionSizes[0] ]
    #     distThs        = [ distThs       [0] ]
    #     distConfs      = [ distConfs     [0] ]

    # Here we hold the results
    # First dimension is class, second overlap
    ap = np.zeros((len(minRegionSizes), len(args.instLabels), len(overlaps)), np.float)

    for dI, minRegionSize in enumerate(minRegionSizes):
        for (oI, overlapTh) in enumerate(overlaps):
            for (lI, labelName) in enumerate(args.instLabels):
                y_true = np.empty(0)
                y_score = np.empty(0)
                # count hard false negatives
                hardFns = 0
                # found at least one gt and predicted instance?
                haveGt = False
                havePred = False

                for img in matches:
                    predInstances = img["prediction"][labelName]
                    gtInstances = img["groundTruth"][labelName]
                    # filter groups in ground truth
                    gtInstances = [
                        gt for gt in gtInstances if gt["pixelCount"] >= minRegionSize
                    ]

                    if gtInstances:
                        haveGt = True
                    if predInstances:
                        havePred = True

                    curTrue = np.ones(len(gtInstances))
                    curScore = np.ones(len(gtInstances)) * (-float("inf"))
                    curMatch = np.zeros(len(gtInstances), dtype=np.bool)

                    # collect matches
                    for (gtI, gt) in enumerate(gtInstances):
                        foundMatch = False
                        for pred in gt["matchedPred"]:
                            overlap = float(pred["maskIntersection"]) / (
                                gt["pixelCount"]
                                + pred["pixelCount"]
                                - pred["maskIntersection"]
                            )
                            if overlap > overlapTh:
                                # the score
                                confidence = pred["confidence"]

                                # if we already hat a prediction for this groundtruth
                                # the prediction with the lower score is automatically a false positive
                                if curMatch[gtI]:
                                    maxScore = max(curScore[gtI], confidence)
                                    minScore = min(curScore[gtI], confidence)
                                    curScore[gtI] = maxScore
                                    # append false positive
                                    curTrue = np.append(curTrue, 0)
                                    curScore = np.append(curScore, minScore)
                                    curMatch = np.append(curMatch, True)
                                # otherwise set score
                                else:
                                    foundMatch = True
                                    curMatch[gtI] = True
                                    curScore[gtI] = confidence

                        if not foundMatch:
                            hardFns += 1

                    # remove non-matched ground truth instances
                    curTrue = curTrue[curMatch == True]
                    curScore = curScore[curMatch == True]

                    # collect non-matched predictions as false positive
                    for pred in predInstances:
                        foundGt = False
                        for gt in pred["matchedGt"]:
                            overlap = float(gt["maskIntersection"]) / (
                                gt["pixelCount"]
                                + pred["pixelCount"]
                                - gt["maskIntersection"]
                            )
                            if overlap > overlapTh:
                                foundGt = True
                                break
                        if not foundGt:
                            # collect number of void and *group pixels
                            nbIgnorePixels = 0
                            for gt in pred["matchedGt"]:
                                # small ground truth instances
                                if gt["pixelCount"] < minRegionSize:
                                    nbIgnorePixels += gt["maskIntersection"]

                            if pred["pixelCount"] <= 0:
                                proportionIgnore = 0
                            else:
                                proportionIgnore = (
                                    float(nbIgnorePixels) / pred["pixelCount"]
                                )
                            # if not ignored
                            # append false positive
                            if proportionIgnore <= overlapTh:
                                curTrue = np.append(curTrue, 0)
                                confidence = pred["confidence"]
                                curScore = np.append(curScore, confidence)

                    # append to overall results
                    y_true = np.append(y_true, curTrue)
                    y_score = np.append(y_score, curScore)

                # compute the average precision
                if haveGt and havePred:
                    # compute precision recall curve first

                    # sorting and cumsum
                    scoreArgSort = np.argsort(y_score)
                    yScoreSorted = y_score[scoreArgSort]
                    yTrueSorted = y_true[scoreArgSort]
                    yTrueSortedCumsum = np.cumsum(yTrueSorted)

                    # unique thresholds
                    (thresholds, uniqueIndices) = np.unique(
                        yScoreSorted, return_index=True
                    )

                    # since we need to add an artificial point to the precision-recall curve
                    # increase its length by 1
                    nbPrecRecall = len(uniqueIndices) + 1

                    # prepare precision recall
                    nbExamples = len(yScoreSorted)
                    nbTrueExamples = yTrueSortedCumsum[-1]
                    precision = np.zeros(nbPrecRecall)
                    recall = np.zeros(nbPrecRecall)

                    # deal with the first point
                    # only thing we need to do, is to append a zero to the cumsum at the end.
                    # an index of -1 uses that zero then
                    yTrueSortedCumsum = np.append(yTrueSortedCumsum, 0)

                    # deal with remaining
                    for idxRes, idxScores in enumerate(uniqueIndices):
                        cumSum = yTrueSortedCumsum[idxScores - 1]
                        tp = nbTrueExamples - cumSum
                        fp = nbExamples - idxScores - tp
                        fn = cumSum + hardFns
                        p = float(tp) / (tp + fp)
                        r = float(tp) / (tp + fn)
                        precision[idxRes] = p
                        recall[idxRes] = r

                    # first point in curve is artificial
                    precision[-1] = 1.0
                    recall[-1] = 0.0

                    # compute average of precision-recall curve
                    # integration is performed via zero order, or equivalently step-wise integration
                    # first compute the widths of each step:
                    # use a convolution with appropriate kernel, manually deal with the boundaries first
                    recallForConv = np.copy(recall)
                    recallForConv = np.append(recallForConv[0], recallForConv)
                    recallForConv = np.append(recallForConv, 0.0)

                    stepWidths = np.convolve(recallForConv, [-0.5, 0, 0.5], "valid")

                    # integrate is now simply a dot product
                    apCurrent = np.dot(precision, stepWidths)

                elif haveGt:
                    apCurrent = 0.0
                else:
                    apCurrent = float("nan")
                ap[dI, lI, oI] = apCurrent

    return ap


def computeAverages(aps, args):
    # max distance index
    # dInf  = np.argmax( args.distanceThs )
    # d50m  = np.where( np.isclose( args.distanceThs ,  50. ) )
    # d100m = np.where( np.isclose( args.distanceThs , 100. ) )
    dInf = np.argmin(args.minRegionSizes)
    o50 = np.where(np.isclose(args.overlaps, 0.5))
    o75 = np.where(np.isclose(args.overlaps, 0.75))

    avgDict = {}
    avgDict["allAp"] = np.nanmean(aps[dInf, :, :])
    avgDict["allAp50%"] = np.nanmean(aps[dInf, :, o50])
    avgDict["allAp75%"] = np.nanmean(aps[dInf, :, o75])

    avgDict["classes"] = {}
    for (lI, labelName) in enumerate(args.instLabels):
        avgDict["classes"][labelName] = {}
        avgDict["classes"][labelName]["ap"] = np.average(aps[dInf, lI, :])
        avgDict["classes"][labelName]["ap50%"] = np.average(aps[dInf, lI, o50])
        avgDict["classes"][labelName]["ap75%"] = np.average(aps[dInf, lI, o75])

    return avgDict


def printResults(avgDict, args):
    strbuffer = io.StringIO()
    # redirect all the print functions to a string buffer
    with redirect_stdout(strbuffer):

        sep = "," if args.csv else ""
        col1 = ":" if not args.csv else ""
        noCol = colors.ENDC if args.colorized else ""
        bold = colors.BOLD if args.colorized else ""
        lineLen = 65

        print("")
        if not args.csv:
            print("#" * lineLen)
        line = bold
        line += "{:<15}".format("what") + sep + col1
        line += "{:>15}".format("AP") + sep
        line += "{:>15}".format("AP_50%") + sep
        line += "{:>15}".format("AP_75%") + sep
        line += noCol
        print(line)
        if not args.csv:
            print("#" * lineLen)

        for (lI, labelName) in enumerate(args.instLabels):
            apAvg = avgDict["classes"][labelName]["ap"]
            ap50o = avgDict["classes"][labelName]["ap50%"]
            ap75o = avgDict["classes"][labelName]["ap75%"]

            line = "{:<15}".format(labelName) + sep + col1
            line += getColorEntry(apAvg, args) + sep + "{:>15.3f}".format(apAvg) + sep
            line += getColorEntry(ap50o, args) + sep + "{:>15.3f}".format(ap50o) + sep
            line += getColorEntry(ap75o, args) + sep + "{:>15.3f}".format(ap75o) + sep
            line += noCol
            print(line)

        allApAvg = avgDict["allAp"]
        allAp50o = avgDict["allAp50%"]
        allAp75o = avgDict["allAp75%"]

        if not args.csv:
            print("-" * lineLen)
        line = "{:<15}".format("average") + sep + col1
        line += getColorEntry(allApAvg, args) + sep + "{:>15.3f}".format(allApAvg) + sep
        line += getColorEntry(allAp50o, args) + sep + "{:>15.3f}".format(allAp50o) + sep
        line += getColorEntry(allAp75o, args) + sep + "{:>15.3f}".format(allAp75o) + sep
        line += noCol
        print(line)
        print("")

        return strbuffer.getvalue()


def prepareJSONDataForResults(avgDict, aps, args):
    JSONData = {}
    JSONData["averages"] = avgDict
    JSONData["overlaps"] = args.overlaps.tolist()
    JSONData["minRegionSizes"] = args.minRegionSizes.tolist()
    JSONData["instLabels"] = args.instLabels
    JSONData["resultApMatrix"] = aps.tolist()

    return JSONData
