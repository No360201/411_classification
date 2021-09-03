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
from core.model_state_dict import densenet_state_dict,model_state_dict
from utils.logger import Logger
from tqdm import tqdm


class classnet_train(class_net_train):
    def __init__(self,config,log_path):
        super().__init__(config)
        self.start_epoch = -1

        if self.resume_model():
            print("resume model!")
        elif config.pretrain.load_pretrained:
            self.load_pretrained(self.config.pretrain)

        self.logger = Logger(logname=log_path, logger="Loss").getlog()

    def load_pretrained(self,config):
        model_dict = self.model.state_dict()
        if self.config.model.arch.find('densenet') >= 0:
            state_dict = densenet_state_dict(config)
        else:
            state_dict = model_state_dict(model_dict,config)

        self.model.load_state_dict(state_dict,strict=False)
        self.model.cuda()


    def resume_model(self):
        if self.config.resume:
            models_path = os.path.join(self.config.work_dir,'models')
            models = os.listdir(models_path)
            models.sort(key=lambda x:int(x[:-4]))
            if len(models):
                latest_checkpoint_path = os.path.join(models_path,models[-1])
                print(latest_checkpoint_path)
                checkpoint = torch.load(latest_checkpoint_path)
                self.model.load_state_dict(checkpoint['net'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                self.start_epoch = checkpoint['epoch']

                return True
        return False


    def train(self):
        print(self.dataset_train.img_datas.class_to_idx)
        print("training")
        for epo in range(self.start_epoch + 1,self.config.epoch):
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
                print("Loss: ",str(loss.item()))

                loss.backward()
                self.optimizer.step()

            print(self.optimizer)
            if epo%5==1:
                model_save_dir = self.config.work_dir + 'models/'
                if not os.path.exists(model_save_dir):
                    os.mkdir(model_save_dir)

                checkpoint = {
                    "net": self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    "epoch": epo,
                    'lr_scheduler': self.lr_scheduler.state_dict()
                }

                torch.save(checkpoint,model_save_dir + "%s.pth" % str(epo))
                # torch.save(self.model.state_dict(), model_save_dir + str(epo)+'.pkl')
                # torch.save(self.model, model_save_dir + str(epo)+'.pth')
            time_end=time.time()
            print(time_end-time_start)
