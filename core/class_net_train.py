import cv2
import os
import numpy as np
from torch.utils import data
from PIL import Image
from torchvision import transforms
from torchvision import datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch.optim import lr_scheduler
import time

from model import build_model
from datasets import build_dataset,build_aug
from utils.optimizer import build_optimizer
from utils.scheduler import get_scheduler
from utils.criterion import build_criterion
from tqdm import tqdm

class class_net_train(object):
    def __init__(self,config):
        self.config=config
        self.creat_dataset(self.config.datasets)        
        self.creat_model(self.config.model)
        self.creat_criterion(self.config.criterion)
        self.cread_optimiter(self.config.optimiter)


    def creat_model(self,config):
        print("creating model")
        self.model=build_model(config)
        if torch.cuda.is_available():
            self.model.cuda()

    def creat_criterion(self,config):
        print("creating criterion")
        self.criterion=build_criterion(config)

    def cread_optimiter(self,config):
        print("creating optimizer")
        self.optimizer = build_optimizer(config,self.model.parameters())
        self.config.lr_scheduler['optimizer'] = self.optimizer
        self.lr_scheduler = get_scheduler(self.config.lr_scheduler)
        
    def creat_dataset(self,config):
        print("creating data")
        self.transform_train,self.transform_test=build_aug(config.transform)
        self.dataset_train,self.dataset_test=build_dataset(config,self.transform_train,self.transform_test)
        self.data_loader=DataLoader(self.dataset_train, batch_size=config.batch_size, shuffle=True)

    def train(self):
        print(self.dataset_train.img_datas.class_to_idx)
        print("training-")
        for epo in range(self.config.epoch):
            self.lr_scheduler.step()
            time_start=time.time()
            for i, data in enumerate(self.data_loader, 0):
                self.model.train()
                inputs, y,_ = data

                inputs = torch.autograd.Variable(inputs)
                y = torch.autograd.Variable(y)
                inputs = inputs.cuda()
                y = y.cuda()

                self.optimizer.zero_grad()
                outputs=self.model(inputs)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
            if epo%5==1:
                torch.save(self.model.state_dict(), self.config.work_dir+str(epo)+'.pkl')
                torch.save(self.model, self.config.work_dir+str(epo)+'.pth')
            time_end=time.time()
            print(time_end-time_start)