import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import json
FOLDER_DATASET = "/home/p_vinsentds/maskrcnn-benchmark/datasets/micr/annotations/"
plt.ion()

class DriveData(Dataset):
    __xs = []
    __ys = []
    
    def __init__(self, folder_dataset, transform=None):
        self.transform = transform
        # Open and load text file including the whole training data
        with open(folder_dataset + "instances_train2017.json", "r") as read_file:
            data = json.load(read_file)


    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        img = Image.open(self.__xs[index])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        # Convert image and label to torch tensors
        img = torch.from_numpy(np.asarray(img))
        label = torch.from_numpy(np.asarray(self.__ys[index]).reshape([1,1]))
        return img, label

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.__xs)

dset_train = DriveData(FOLDER_DATASET)
train_loader = DataLoader(dset_train, batch_size=10, shuffle=True, num_workers=1)