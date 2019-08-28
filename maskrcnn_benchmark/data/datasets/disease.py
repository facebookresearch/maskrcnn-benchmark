import torch

from maskrcnn_benchmark.structures.bounding_box import BoxList
from . import build_disease as bd


# TODO Review readme files and make sure to add this dataset to the necessary functions


class DiseaseDataset(object):
    def __init__(self, json_name='disease_data_fp_7_23_2019', dataroot='../raw', transforms=None):
        self.dataroot = dataroot
        self.json_name = json_name
        self.transforms = transforms

        df, label_types = bd.create_disease_dataset(self.json_name, self.dataroot)
        class_names = [name.strip().lower().replace(' ', '_') for name in label_types]

        self.class_names = class_names
        self.dataframe = df

    def __getitem__(self, idx):
        # load the image as a PIL Image
        image = Image.open(self.dataframe['image_path'].iat[idx]).convert('RGB')

        # load the bounding boxes as a list of list of boxes
        # in this case, for illustrative purposes, we use
        # x1, y1, x2, y2 order.
        # boxes = [[0, 0, 10, 10], [10, 20, 50, 50]]
        boxes = self.dataframe['all_boxes'].iat[idx]
        # and labels
        # labels = torch.tensor([10, 20])
        labels = torch.tensor(self.dataframe['all_labels'].iat[idx])

        # create a BoxList from the boxes
        boxlist = BoxList(boxes, image.size, mode="xyxy")
        # add the labels to the boxlist
        boxlist.add_field("labels", labels)

        if self.transforms:
            image, boxlist = self.transforms(image, boxlist)

        # return the image, the boxlist and the idx in your dataset
        return image, boxlist, idx

    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk

        # TODO Should look at image size without actually loading it
        image = Image.open(self.dataframe['image_path'].iat[idx]).convert('RGB')

        return {"height": image.shape[0], "width": image.shape[1]}
