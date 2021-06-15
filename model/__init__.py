from model.net import *
from model.net_test import *

__all__=['ResNet','resnet34', 'resnet50', 'resnet101', 'resnet152','SENet','se_resnet_34', 'se_resnet_50'
           ,'resnext101','resnext50','resnext152','resnet34_fpn', 'resnet50_fpn']

def build_model(config):
    return globals()[config['arch']](**config['kwargs'])