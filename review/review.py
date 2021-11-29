#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    https://kazuto1011.github.io
# Date:   05 February 2019


from __future__ import print_function

import yaml
from easydict import EasyDict
import click
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import models, transforms

from grad_cam import GradCAM,GuidedBackPropagation
from model import build_model

class visualize_grad_cam():
    def __init__(self,config_path,cuda):
        self.get_device(cuda)
        self.load_config(config_path)
        self.load_model()
        self.load_image()
        self.gcam = GradCAM(self.model)

        self.model_name = self.config['model']['arch']

    def get_device(self,cuda):
        self.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
        if cuda:
            current_device = torch.cuda.current_device()
            print("Running on the GPU:", torch.cuda.get_device_name(current_device))
        else:
            print("Running on the CPU")

    def load_config(self,config_path):
        with open(config_path, 'rb') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.image_size = self.config['datasets']['transform']['train']['kwargs']['size']
        config = self.config['review']
        self.model_path = config['model_path']
        self.image_path = config['image_path']
        self.target_layers = config['visualize_layers']

    def load_model(self):
        self.model = build_model(self.config['model'])
        model_dict = torch.load(self.model_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(model_dict['net'])
        self.model.to(self.device)
        self.model.eval()

    def load_image(self):
        raw_image = cv2.imread(self.image_path)[..., ::-1]
        self.raw_image = cv2.resize(raw_image, (self.image_size[0],) * 2)
        image = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )(self.raw_image).unsqueeze(0)
        self.image = image.to(self.device)

    def run_gcam(self):
        predictions = self.gcam.forward(self.image)
        top_idx = predictions[0][1]
        for target_layer in self.target_layers:
            self.gcam.backward(idx=top_idx)
            region = self.gcam.generate(target_layer=target_layer)
            img_name = "{}-gradcam-{}-{}.png".format(self.model_name, target_layer,top_idx)
            self.save_gradcam(img_name,region)

    def save_gradcam(self,filename,region):
        h, w, _ = self.raw_image.shape
        gcam = cv2.resize(region, (w, h))
        gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
        gcam = gcam.astype(np.float) + self.raw_image.astype(np.float)
        gcam = gcam / gcam.max() * 255.0
        cv2.imwrite(filename, np.uint8(gcam))


@click.command()
@click.option("-c", "--config-path", type=str, required=True,default="../experiment/test/config.yaml")
@click.option("--cuda/--no-cuda", default=True)
def main(config_path,cuda):
    gcam_func = visualize_grad_cam(config_path,cuda)
    gcam_func.run_gcam()

    # device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    #
    # if cuda:
    #     current_device = torch.cuda.current_device()
    #     print("Running on the GPU:", torch.cuda.get_device_name(current_device))
    # else:
    #     print("Running on the CPU")
    # #
    # # # Synset words
    # # classes = list()
    # # with open("samples/synset_words.txt") as lines:
    # #     for line in lines:
    # #         line = line.strip().split(" ", 1)[1]
    # #         line = line.split(", ", 1)[0].replace(" ", "_")
    # #         classes.append(line)
    #
    # # load config
    # with open(config_path,'rb') as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    # config = EasyDict(config)
    #
    # # Model
    # model = build_model(config)
    # model_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    # model.load_state_dict(model_dict['net'])
    # model.to(device)
    # model.eval()
    #
    # # Image
    # raw_image = cv2.imread(image_path)[..., ::-1]
    # raw_image = cv2.resize(raw_image, (224,) * 2)
    # image = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ]
    # )(raw_image).unsqueeze(0)
    # image = image.to(device)
    #
    # gcam = GradCAM(model=model)
    # predictions = gcam.forward(image)
    # top_idx = predictions[0][1]
    #
    # # guided_bp = GuidedBackPropagation(model=model)
    # # predictions = guided_bp.forward(image)
    # #
    # # guided_bp.backward(idx=top_idx)
    # # image_grad = guided_bp.generate()
    #
    #
    # for target_layer in ["layer1","layer2","layer3","layer4"]:
    #     print("Generating Grad-CAM @{}".format(target_layer))
    #
    #     # Grad-CAM
    #     gcam.backward(idx=top_idx)
    #     region = gcam.generate(target_layer=target_layer)
    #
    #     # save_gradcam(
    #     #     "results/{}-gradcam-{}-{}.png".format(
    #     #         "resnet152", target_layer, classes[top_idx]
    #     #     ),
    #     #     region,
    #     #     raw_image,
    #     # )
    #     save_gradcam(
    #         "{}-gradcam-{}-{}.png".format(
    #             "resnet152", target_layer, top_idx
    #         ),
    #         region,
    #         image_grad,
    #         raw_image,
    #     )


if __name__ == "__main__":
    main()
