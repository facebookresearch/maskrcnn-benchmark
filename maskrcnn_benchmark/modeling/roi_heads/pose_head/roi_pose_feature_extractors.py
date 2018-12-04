from ..box_head.roi_box_feature_extractors import ResNet50Conv5ROIFeatureExtractor

def make_roi_pose_feature_extractor(cfg):
    return ResNet50Conv5ROIFeatureExtractor(cfg)
