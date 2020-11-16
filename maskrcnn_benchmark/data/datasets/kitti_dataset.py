import torch
import torchvision
import os
import sys
import PIL.Image as Image
import cv2
import array
import glob
import numpy as np
import maskrcnn_benchmark.utils.object3d_kitti as object3d_kitti
import maskrcnn_benchmark.utils.calibration_kitti as calibration_kitti
from maskrcnn_benchmark.structures.bounding_box import BoxList

sys.path.append(os.getcwd())

class KittiDataset(torch.utils.data.Dataset):
    """
    For kitti dataset
    """
    def __init__(self, data_root, list_file, transforms, is_train):
        self.dims_avg = {'Cyclist': np.array([ 1.73532436,  0.58028152,  1.77413709]), 'Van': np.array([ 2.18928571,  1.90979592,  5.07087755]), 'Tram': np.array([  3.56092896,   2.39601093,  18.34125683]), 'Car': np.array([ 1.52159147,  1.64443089,  3.85813679]), 'Pedestrian': np.array([ 1.75554637,  0.66860882,  0.87623049]), 'Truck': np.array([  3.07392252,   2.63079903,  11.2190799 ])}
        self.BIN = 2
        self.overlap = 0.1

        self.is_train = is_train
        self.transforms = transforms
        self.root_split_path = data_root
        self.list_file = list_file

        self.points_clouds_dirs = []
        self.image_dirs = []
        self.calib = []
        self.label_dirs = []
        self.sample_id_list = [x.strip() for x in open(self.list_file).readlines()]

    def get_lidar(self, idx):
        lidar_file = self.root_split_path + '/velodyne/' + ('%s.bin' % idx)
        assert os.path.join(lidar_file)
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_image(self, idx):
        img_file = self.root_split_path + '/image_2/' + ('%s.png' % idx)
        assert os.path.join(img_file)
        return Image.open(img_file).convert("RGB")

    def get_label(self, idx):
        label_file = self.root_split_path + '/label_2/' + ('%s.txt' % idx)
        assert os.path.join(label_file)
        return object3d_kitti.get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = self.root_split_path + '/calib/' + ('%s.txt' % idx)
        assert os.path.join(calib_file)
        return calibration_kitti.Calibration(calib_file)
    
    def compute_anchors(self, angle, bin, overlap):
        anchors = []
        
        wedge = 2. * np.pi / bin
        l_index = int(angle / wedge)
        r_index = l_index + 1
        
        if (angle - l_index * wedge) < wedge / 2 * (1 + overlap/2):
            anchors.append([l_index, angle - l_index * wedge])
            
        if (r_index*wedge - angle) < wedge / 2 * (1 + overlap / 2):
            anchors.append([r_index % bin, angle - r_index * wedge])
            
        return anchors  

    def __len__(self):
        return len(self.sample_id_list)

    def __getitem__(self, item):
        index = self.sample_id_list[item]

        #pcd_lidar = self.get_lidar(index)
        image = self.get_image(index)
        image_size = image.size
        # print(image.shape)
        labels = self.get_label(index)
        calib = self.get_calib(index)

        boxes = np.array([label.box2d for label in labels if label.get_kitti_obj_level() >= 0 and label.cls_id > -1])
        classes = np.array([label.cls_id for label in labels if label.get_kitti_obj_level() >= 0 and label.cls_id > -1])
        alphas = np.array([label.alpha for label in labels if label.get_kitti_obj_level() >= 0 and label.cls_id > -1])
        locations = np.array([label.loc for label in labels if label.get_kitti_obj_level() >= 0 and label.cls_id > -1])
        dimensions = np.array([[label.h, label.w, label.l] for label in labels if label.get_kitti_obj_level() >= 0 and label.cls_id > -1])
        relative_dimensions = np.array([[label.h - self.dims_avg[label.cls_type][0], label.w - self.dims_avg[label.cls_type][1], label.l - self.dims_avg[label.cls_type][2]] for label in labels if label.get_kitti_obj_level() >= 0 and label.cls_id > -1])
        rotation_y = np.array([label.ry for label in labels if label.get_kitti_obj_level() >= 0 and label.cls_id > -1])

        # pos_ids = np.where(classes > -1)
        # boxes = boxes[pos_ids]
        # classes = classes[pos_ids]
        # alphas = alphas[pos_ids]
        # locations = locations[pos_ids]
        # dimensions = dimensions[pos_ids]
        # relative_dimensions = relative_dimensions[pos_ids]
        # rotation_y = rotation_y[pos_ids]

        alpha_conf = []
        alpha_oriention = []
        for alp in alphas:
            # Fix orientation and confidence for no flip
            orientation = np.zeros((self.BIN, 2))
            confidence = np.zeros(self.BIN)

            anchors = compute_anchors(alp, self.BIN. self.overlap)

            for anchor in anchors:
                orientation[anchor[0]] = np.array([np.cos(anchor[1]), np.sin(anchor[1])])
                confidence[anchor[0]] = 1.

            confidence = confidence / np.sum(confidence)

            alpha_conf.append(confidence)
            alpha_oriention.append(orientation)

        boxes = torch.as_tensor(boxes).reshape(-1, 4)
        classes = torch.as_tensor(classes).reshape(-1)
        alphas = torch.as_tensor(alphas).reshape(-1, 1)
        locations = torch.as_tensor(locations).reshape(-1, 3)
        dimensions = torch.as_tensor(dimensions).reshape(-1, 3)
        relative_dimensions = torch.as_tensor(relative_dimensions).reshape(-1, 3)
        rotation_y = torch.as_tensor(rotation_y).reshape(-1, 1)
        alpha_conf = torch.as_tensor(alpha_conf).reshape(-1, self.BIN)
        alpha_oriention = torch.as_tensor(alpha_oriention).reshape(-1, self.BIN, 2)

        target = BoxList(boxes, image_size, mode='xyxy')
        target.add_field('labels', classes)
        target.add_field('alphas', alphas)
        target.add_field('locations', locations)
        target.add_field('dimensions', dimensions)
        target.add_field('relative_dimensions', relative_dimensions)
        target.add_field('rotation_y', rotation_y)
        target.add_field('alpha_conf', alpha_conf)
        target.add_field('alpha_oriention', alpha_oriention)
        #target.add_field('calib', calib)

        if self.transforms is not None:
            image, target = self.transforms(image, target)
            if len(target) == 0:
                if item == len(self) - 1:
                    next = item - 1
                else:
                    next = item + 1
                return self[next]

        return image, target, item

    def get_img_info(self, item):
        index = self.sample_id_list[item]
        image = self.get_image(index)
        img_info = {"height": image.size[1], "width": image.size[0]}
        return img_info


def viz_kitti(batch):
    image, targets, item = batch
    image = cv2.cvtColor(numpy.asarray(image),cv2.COLOR_RGB2BGR)
    print(image.shape)
    img1 = image.copy()
    img2 = image.copy()

    bboxes = targets.bbox
    cls_ids = targets.get_field("labels").cpu().numpy()
    locations = targets.get_field("locations").cpu().numpy()
    dimensions = targets.get_field("dimensions").cpu().numpy()
    rotation_y = targets.get_field("rotation_y").cpu().numpy()
    calib = targets.get_field("calib")

    for i, bbox in enumerate(bboxes):
        xmin, ymin, xmax, ymax = bbox

        cls_id = cls_ids[i]
        x, y, z = locations[i]
        h, w, l = dimensions[i]
        x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
        y_corners = [0,0,0,0,-h,-h,-h,-h]
        z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
        corners_3d = np.vstack([x_corners, y_corners, z_corners])

        c = np.cos(rotation_y[i])
        s = np.sin(rotation_y[i])
        R = np.array([c, 0, s, 0, 1, 0, -s, 0, c]).astype(np.float32).reshape(3, 3)
        corners_3d = np.dot(R, corners_3d)
        corners_3d[0, :] = corners_3d[0, :] + x
        corners_3d[1, :] = corners_3d[1, :] + y
        corners_3d[2, :] = corners_3d[2, :] + z

        if np.any(corners_3d[2,:]<0.1):
            continue

        n = corners_3d.shape[1]
        corners_3d_h = np.vstack((corners_3d, np.ones((1, n))))
        pts_2d = np.dot(calib.P2, corners_3d_h) # 3xn
        pts_2d[0, :] /= pts_2d[2, :]
        pts_2d[1, :] /= pts_2d[2, :]
        pts_2d = np.transpose(pts_2d)

        if cls_id != -1:
            color = (255, 0, 0)
            thickness = 1
            cv2.rectangle(img1, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
            qs = pts_2d.astype(np.int32)
            for k in range(0,4):
                i,j=k,(k+1)%4
                # use LINE_AA for opencv3
                cv2.line(img2, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness)

                i,j=k+4,(k+1)%4 + 4
                cv2.line(img2, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness)

                i,j=k,k+4
                cv2.line(img2, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness)
        else:
            color = (255, 255, 0)
            thickness = 1
            cv2.rectangle(img1, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 0), 2)
            qs = pts_2d.astype(np.int32)
            for k in range(0,4):
                i,j=k,(k+1)%4
                # use LINE_AA for opencv3
                cv2.line(img2, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness)

                i,j=k+4,(k+1)%4 + 4
                cv2.line(img2, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness)

                i,j=k,k+4
                cv2.line(img2, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness)

    #cv2.imshow("img1", img1)
    #cv2.imshow("img2", img2)
    #cv2.waitKey()

    cv2.imwrite("img1.jpg", img1)
    cv2.imwrite("img2.jpg", img2)


if __name__ == "__main__":
    #test KittiDataset
    dataset = KittiDataset("datasets/kitti/training", "datasets/kitti/ImageSet/train.txt", transforms=None, is_train=True)
    for idx, batch in enumerate(dataset):
        viz_kitti(batch)

