import os
from torch import optim
import torch.nn as nn
from torch.optim import lr_scheduler

def build_optimizer(config,param):
    if config['type']=="SGD":
        return optim.SGD(param,**config['kwargs'])
    elif config['type']=="Adam":
        return optim.Adam(param, **config['kwargs'])