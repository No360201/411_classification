from datasets.dataset import *
from datasets.data_aug import *

__all__=['transform_train','transform_test','VGGDataset']

def build_aug(config):
    transform_train=globals()[config.train['type']](**config.train['kwargs'])
    transform_test=globals()[config.test['type']](**config.train['kwargs'])
    return transform_train,transform_test

def build_dataset(config,transform_train,transform_test):
    train_dataset=globals()[config.train['type']](**config.train['kwargs'],transform_data=transform_train)
    test_dataset=globals()[config.test['type']](**config.test['kwargs'],transform_data_test=transform_test)
    return train_dataset,test_dataset
