import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from torch.utils import data
from PIL import Image
from torchvision import transforms
from torchvision import datasets
import torch as t
from torch.utils.data import Dataset, DataLoader
from datasets.data_aug import *

class VGGDataset(Dataset):
    def __init__(self, train_path=None,test_path=None,train=True,transform_data=None,transform_data_test=None):
        if train:
            self.img_datas = datasets.ImageFolder(train_path)
            self.transform = transform_data
            # print (self.img_datas.class_to_idx)
        else:
            self.img_val = datasets.ImageFolder(test_path)
            self.transform = transform_data_test
        self.train = train

    def __getitem__(self, idx):
        if self.train:
            imgA = cv2.imread(self.img_datas.imgs[idx][0])
            label = self.img_datas.imgs[idx][1]
            img_name=self.img_datas.imgs[idx][0]
            imgA = self.transform(imgA)
            imgA = imgA.cuda()
        else:
            imgA = cv2.imread(self.img_val.imgs[idx][0])
            label = self.img_val.imgs[idx][1]
            imgA = self.transform(imgA)
            imgA = imgA.cuda()
            img_name = self.img_val.imgs[idx][0]
        return imgA, label, img_name

    def __len__(self):
        if self.train:
            return len(self.img_datas.imgs)
        else:
            return len(self.img_val.imgs)

