# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import pickle
from collections import OrderedDict

import torch


def _rename_weights_for_R50(weights):
    original_keys = sorted(weights.keys())
    layer_keys = sorted(weights.keys())
    layer_keys = [k.replace("_", ".") for k in layer_keys]
    layer_keys = [k.replace(".w", ".weight") for k in layer_keys]
    layer_keys = [k.replace(".bn", "_bn") for k in layer_keys]
    layer_keys = [k.replace(".b", ".bias") for k in layer_keys]
    layer_keys = [k.replace("_bn.s", "_bn.scale") for k in layer_keys]
    layer_keys = [k.replace(".biasranch", ".branch") for k in layer_keys]
    layer_keys = [k.replace("bbox.pred", "bbox_pred") for k in layer_keys]
    layer_keys = [k.replace("cls.score", "cls_score") for k in layer_keys]
    layer_keys = [k.replace("res.conv1_", "conv1_") for k in layer_keys]

    # RPN / Faster RCNN
    layer_keys = [k.replace(".biasbox", ".bbox") for k in layer_keys]
    layer_keys = [k.replace("conv.rpn", "rpn.conv") for k in layer_keys]
    layer_keys = [k.replace("rpn.bbox.pred", "rpn.bbox_pred") for k in layer_keys]
    layer_keys = [k.replace("rpn.cls.logits", "rpn.cls_logits") for k in layer_keys]

    # FPN
    layer_keys = [
        k.replace("fpn.inner.res2.2.sum.lateral", "fpn_inner1") for k in layer_keys
    ]
    layer_keys = [
        k.replace("fpn.inner.res3.3.sum.lateral", "fpn_inner2") for k in layer_keys
    ]
    layer_keys = [
        k.replace("fpn.inner.res4.5.sum.lateral", "fpn_inner3") for k in layer_keys
    ]
    layer_keys = [k.replace("fpn.inner.res5.2.sum", "fpn_inner4") for k in layer_keys]

    layer_keys = [k.replace("fpn.res2.2.sum", "fpn_layer1") for k in layer_keys]
    layer_keys = [k.replace("fpn.res3.3.sum", "fpn_layer2") for k in layer_keys]
    layer_keys = [k.replace("fpn.res4.5.sum", "fpn_layer3") for k in layer_keys]
    layer_keys = [k.replace("fpn.res5.2.sum", "fpn_layer4") for k in layer_keys]

    layer_keys = [k.replace("rpn.conv.fpn2", "rpn.conv") for k in layer_keys]
    layer_keys = [k.replace("rpn.bbox_pred.fpn2", "rpn.bbox_pred") for k in layer_keys]
    layer_keys = [
        k.replace("rpn.cls_logits.fpn2", "rpn.cls_logits") for k in layer_keys
    ]

    # Mask R-CNN
    layer_keys = [k.replace("mask.fcn.logits", "mask_fcn_logits") for k in layer_keys]
    layer_keys = [k.replace(".[mask].fcn", "mask_fcn") for k in layer_keys]
    layer_keys = [k.replace("conv5.mask", "conv5_mask") for k in layer_keys]

    # Keypoint R-CNN
    layer_keys = [k.replace("kps.score.lowres", "kps_score_lowres") for k in layer_keys]
    layer_keys = [k.replace("kps.score", "kps_score") for k in layer_keys]
    layer_keys = [k.replace("conv.fcn", "conv_fcn") for k in layer_keys]

    # from IPython import embed; embed()

    key_map = {k: v for k, v in zip(original_keys, layer_keys)}

    new_weights = OrderedDict()
    for k, v in weights.items():
        if "_momentum" in k:
            continue
        # if 'fc1000' in k:
        #     continue
        # new_weights[key_map[k]] = torch.from_numpy(v)
        w = torch.from_numpy(v)
        if "bn" in k:
            w = w.view(1, -1, 1, 1)
        new_weights[key_map[k]] = w

    return new_weights


def _load_c2_pickled_weights(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    if "blobs" in data:
        weights = data["blobs"]
    else:
        weights = data
    return weights


def _get_rpn_state_dict(state_dict):
    rpn_state_dict = OrderedDict()
    rpn_mapping = {
        "rpn.conv.weight": "conv.weight",
        "rpn.conv.bias": "conv.bias",
        "rpn.cls_logits.weight": "cls_logits.weight",
        "rpn.cls_logits.bias": "cls_logits.bias",
        "rpn.bbox_pred.weight": "bbox_pred.weight",
        "rpn.bbox_pred.bias": "bbox_pred.bias",
    }
    for k, v in rpn_mapping.items():
        rpn_state_dict[v] = state_dict[k]

    return rpn_state_dict


def load_c2_weights_faster_rcnn_resnet50_c4(model, file_path):
    state_dict = _load_c2_pickled_weights(file_path)
    state_dict = _rename_weights_for_R50(state_dict)

    rpn_state_dict = _get_rpn_state_dict(state_dict)

    model.backbone.stem.load_state_dict(state_dict, strict=False)
    model.backbone.load_state_dict(state_dict, strict=False)
    model.rpn.heads.load_state_dict(rpn_state_dict)

    model.roi_heads.heads[0].feature_extractor.head.load_state_dict(
        state_dict, strict=False
    )
    model.roi_heads.heads[0].predictor.load_state_dict(state_dict, strict=False)


def load_c2_weights_faster_rcnn_resnet50_fpn(model, file_path):
    state_dict = _load_c2_pickled_weights(file_path)
    state_dict = _rename_weights_for_R50(state_dict)

    rpn_state_dict = _get_rpn_state_dict(state_dict)

    model.backbone[0].stem.load_state_dict(state_dict, strict=False)
    model.backbone[0].load_state_dict(state_dict, strict=False)
    # FPN
    model.backbone[1].load_state_dict(state_dict, strict=False)

    model.rpn.heads.load_state_dict(rpn_state_dict)

    model.roi_heads.heads[0].feature_extractor.load_state_dict(state_dict, strict=False)
    model.roi_heads.heads[0].predictor.load_state_dict(state_dict, strict=False)


def load_c2_weights_mask_rcnn_resnet50_c4(model, file_path):
    state_dict = _load_c2_pickled_weights(file_path)
    state_dict = _rename_weights_for_R50(state_dict)

    rpn_state_dict = _get_rpn_state_dict(state_dict)

    model.backbone.stem.load_state_dict(state_dict, strict=False)
    model.backbone.load_state_dict(state_dict, strict=False)
    model.rpn.heads.load_state_dict(rpn_state_dict)

    model.roi_heads.heads[0].feature_extractor.head.load_state_dict(
        state_dict, strict=False
    )
    model.roi_heads.heads[0].predictor.load_state_dict(state_dict, strict=False)

    model.roi_heads.heads[1].predictor.load_state_dict(state_dict, strict=False)


def load_c2_weights_mask_rcnn_resnet50_fpn(model, file_path):
    state_dict = _load_c2_pickled_weights(file_path)
    state_dict = _rename_weights_for_R50(state_dict)

    rpn_state_dict = _get_rpn_state_dict(state_dict)

    model.backbone[0].stem.load_state_dict(state_dict, strict=False)
    model.backbone[0].load_state_dict(state_dict, strict=False)
    # FPN
    model.backbone[1].load_state_dict(state_dict, strict=False)

    model.rpn.heads.load_state_dict(rpn_state_dict)

    model.roi_heads.heads[0].feature_extractor.load_state_dict(state_dict, strict=False)
    model.roi_heads.heads[0].predictor.load_state_dict(state_dict, strict=False)

    model.roi_heads.heads[1].feature_extractor.load_state_dict(state_dict, strict=False)
    model.roi_heads.heads[1].predictor.load_state_dict(state_dict, strict=False)


_C2_WEIGHT_LOADER = {
    "faster_rcnn_R_50_C4": load_c2_weights_faster_rcnn_resnet50_c4,
    "faster_rcnn_R_50_FPN": load_c2_weights_faster_rcnn_resnet50_fpn,
    "mask_rcnn_R_50_C4": load_c2_weights_mask_rcnn_resnet50_c4,
    "mask_rcnn_R_50_FPN": load_c2_weights_mask_rcnn_resnet50_fpn,
}


def load_from_c2(cfg, model, weights_file):
    loader_name = cfg.MODEL.C2_COMPAT.WEIGHT_LOADER
    loader = _C2_WEIGHT_LOADER[loader_name]
    loader(model, cfg.MODEL.C2_COMPAT.WEIGHTS)
