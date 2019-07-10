# location of pycoco tools,
# /home/p_vinsentds/.conda/envs/qanet/lib/python3.6/site-packages/pycocotools-2.0-py3.6-linux-x86_64.egg/pycocotools
#   lists = [self.imgToAnns[str(imgId)] for imgId in imgIds if str(imgId) in self.imgToAnns] in pycoco tools/coco.py file
import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints
from .coco import COCODataset
from PIL import Image

min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False

DATA_DIR = "/home/p_vinsentds/maskrcnn-benchmark/datasets/micr/"
folder = "train2017"
img_path   = DATA_DIR + folder
# print("you have reached micr datatset get method")
class MICRDataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super().__init__(self, ann_file,root)
        # super(MICRDataset, self).__init__(root,ann_file)
        # sort indices for reproducible results
        from pycocotools.coco import COCO
        self.coco = COCO(ann_file)
        self.ids = sorted(self.ids)
        print(f"printing ids {self.ids}")
        # # filter images without detection annotations
        if remove_images_without_annotations:
            print("remove_images_without_annotations")
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    # print(f"has valid annotations {anno}")
                    ids.append(img_id)
            self.ids = ids
            # print(f"has valid ids {self.ids}")

        self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()} # for cat in self.coco.cats.values()
        print(f"self.categories:{self.categories}")
        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        print(f"self.json_category_id_to_contiguous_id:{self.json_category_id_to_contiguous_id}")
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        print(f"self.contiguous_category_id_to_json_id:{self.contiguous_category_id_to_json_id}")
       
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

        print(f"self.id_to_img_map:{self.id_to_img_map}")
        print(f"self._transforms:{self._transforms}")


    def __getitem__(self, idx):

        
        # img, anno = super(MICRDataset, self).__getitem__(idx) # TODO changed from MICRDataset to COCODataset # super(MICRDataset, self)
        img =  Image.open(img_path + idx + '.jpg').convert("RGB")
        print(img)
        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == "0"] #TODO need to check the for type as string
        print("anno!!!!!!!!!")
        print(anno)
        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")
        import pdb;pdb.set_trace()
        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)
        # target.add_field("labels", classes)
        if anno and "segmentation" in anno[0]:
            masks = [obj["segmentation"] for obj in anno]
            masks = SegmentationMask(masks, img.size, mode='poly')
            target.add_field("masks", masks)

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data


# TODO change the file numbers in train and test file or include only 15 files in train and rest in test and val...folder
# TODO put a trace in pycoco coco.py ...file also...
