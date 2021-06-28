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
from core.class_net_train import class_net_train
from utils.logger import Logger
from tqdm import tqdm


class classnet_train(class_net_train):
    def __init__(self,config,log_path):
        super().__init__(config)
        self.logger = Logger(logname=log_path, logger="Loss").getlog()



    def train(self):
        print(self.dataset_train.img_datas.class_to_idx)
        print("training")
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

                if epo == 0 and i == 0:
                    self.logger.info("Train config: ")
                    for key,val in self.config.items():
                        if isinstance(val,dict):
                            self.logger.info(str(key) + " : ")
                            for _key,_val in val.items():
                                self.logger.info("\t" + str(_key) + " : " + str(_val))
                        else:
                            self.logger.info(str(key) + " : " + str(val))
                self.logger.info("Loss: epoch: " + str(epo) + " iter_num: " + str(i) + " loss: " + str(loss.item()))

                loss.backward()
                self.optimizer.step()
            if epo%5==1:
                model_save_dir = self.config.work_dir + 'models/'
                if not os.path.exists(model_save_dir):
                    os.mkdir(model_save_dir)

                torch.save(self.model.state_dict(), model_save_dir + str(epo)+'.pkl')
                torch.save(self.model, model_save_dir + str(epo)+'.pth')
            time_end=time.time()
            print(time_end-time_start)
