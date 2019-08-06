# Abstract dataset definition for custom datasets
# by botcs@github

import torch


class AbstractDataset(torch.utils.data.Dataset):
    """
    Serves as a common interface to reduce boilerplate and help dataset
    customization

    A generic Dataset for the maskrcnn_benchmark must have the following
    non-trivial fields / methods implemented:
        classid_to_name - dict:
            This will allow the trivial generation of classid_to_ccid 
            (contiguous) and ccid_to_classid (reversed)

        __getitem__ - function(idx):
            This has to return three things: img, target, idx.
            img is the input image, which has to be load as a PIL Image object
            implementing the target requires the most effort, since it must have
            multiple fields: the size, bounding boxes, labels (contiguous), and
            masks (either COCO-style Polygons, RLE or torch BinaryMask).
            Ideally the target is a BoxList instance with extra fields.
            Lastly, idx is simply the input argument of the function.

    also the following is required:
        __len__ - function():
            return the size of the dataset
        get_img_info - function(idx):
            return metadata, at least width and height of the input image
    """

    def __init__(self, *args, **kwargs):
        self.classid_to_name = None
        self.classid_to_ccid = None

        self.name_to_classid = None
        self.name_to_ccid = None

        self.ccid_to_classid = None
        self.ccid_to_name = None

    def __getitem__(self, idx):
        raise NotImplementedError

    def initMaps(self):
        """
        it is required to have classid->name mapping to start with

        Initialize mappings between:
            classid <-> name <-> continuous class id (ccid)

        classid: originally in COCO this goes from 1 to 90 and not necessarily 
            continuously

        name: this is a string that is linked to the classid

        ccid: positive int represent the MaskRCNN heads. Must be continuous


        NOTE:
        It is important that classid should not list the background
        more important, that ccid must start from 1, since the backround is
        always associated to ccid=0 by the framework.
        """

        assert self.classid_to_name is not None
        self.classid_to_ccid = {
            classid: ccid
            for ccid, classid in enumerate(self.classid_to_name.keys(), 1)
        }

        self.name_to_classid = {
            name: classid for classid, name in self.classid_to_name.items()
        }
        self.name_to_ccid = {
            name: self.classid_to_ccid[classid]
            for classid, name in self.classid_to_name.items()
        }

        self.ccid_to_classid = {
            ccid: classid for classid, ccid in self.classid_to_ccid.items()
        }
        self.ccid_to_name = {
            ccid: name for name, ccid in self.name_to_ccid.items()
        }

    def get_img_info(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
