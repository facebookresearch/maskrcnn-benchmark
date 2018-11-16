"""
A simple demo for extracting the encoding features of object bounding boxes
in images and store it on an HDF5 file.
This features can be used as bottom-up attention features in Image captioning described at [here](https://arxiv.org/abs/1707.07998)
"""
import cv2
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import os
from glob import glob
import argparse
from math import ceil
import h5py


join_path=lambda x: os.path.join(os.path.dirname(__file__),x)
config_file = join_path("../configs/caffe2/e2e_faster_rcnn_X_101_32x8d_FPN_1x_caffe2.yaml")

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)

class ImageLoader(object):
    """
    A simple class for batching images to extract encoded features.
    """
    def __init__(self, base_path, extension='jpg', batch_size=32):
        self._base_path = base_path
        self._images_path = glob(os.path.join(self._base_path, "*."+extension))
        self._batch_size=batch_size
        self._num_images=len(self._images_path)
        self._total_batches=ceil(self._num_images/self._batch_size)
        print("Number of images for object encoding: %d"%self._num_images)
        if self._num_images == 0:
            raise ValueError("No images exists in the base path. Maybe you had provided wrong path.")
    
    @property
    def num_images(self):
        return self._num_images

    def __len__(self):
        return len(self._images_path)
    
    def __getitem__(self, index):
        return cv2.imread(self._images_path[index])
    
    def next_batch(self):
        for i in range(self._total_batches):
            batch_slice=range(i*self._batch_size,(i+1)*self._batch_size)
            batch = [self[i] for i in batch_slice]
            names = [self._images_path[i] for i in batch_slice]
            yield batch, names

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", "--base_path", help="Base path for images", required=True)
    parser.add_argument("--extension", help="Extensions of images", default="jpg")
    parser.add_argument("--batch_size", help="Batch size", default=5, type=int)
    parser.add_argument("--max_num_bbox", help="Maximum number of object bounding boxes for each image", default=36, type=int)
    parser.add_argument("--storing_path", help="Storing path of hdf5 file", default="encoding_features.h5")
    encoding_dim=1024
    args=parser.parse_args()
    h_stores=h5py.File(args.storing_path, 'w')
    image_loader=ImageLoader(args.base_path, extension=args.extension, batch_size=args.batch_size)
    encoding_features=h_stores.create_dataset("encoding_features",shape=(image_loader.num_images, args.max_num_bbox, encoding_dim), dtype='f')
    bboxes=h_stores.create_dataset("bboxes", shape=(image_loader.num_images, args.max_num_bbox,4), dtype='f')
    image_names=h_stores.create_dataset("image_names", shape=(image_loader.num_images,), dtype=h5py.special_dtype(vlen=str))

    index=0
    for batch, names in image_loader.next_batch():
        predictions = coco_demo.extract_encoding_features(batch)
        for i,p in enumerate(predictions):
            m = min(args.max_num_bbox, len(p.extra_fields['encoding_features']))
            encoding_features[index,:m,:]=p.extra_fields['encoding_features'].numpy()[:m,:]
            bboxes[index,:m-1,:]=p.bbox.numpy()[:m-1,:]
            image_names[index]=names[i].split('/')[-1]
            index+=1
    h_stores.close()

if __name__ == '__main__':
    main()
