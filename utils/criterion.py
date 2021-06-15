import os
import torch.nn as nn

def build_criterion(config):
    return globals()[config['type']](**config['kwargs'])

class Criterion(nn.Module):
    def __init__(self,type):
        super(Criterion,self).__init__()
        if type=="CrossEntropy":
            self.criterion=nn.CrossEntropyLoss()
    def forward(self,y,target):
        return self.criterion(y,target)