from train import classnet_train
from test import test_multiple_models

# from deploy.torch2onnx import torch2onnx
# from deploy.calib import CenterNetEntropyCalibrator
# from deploy.onnxtrt import get_engine,run_trt


from easydict import EasyDict
import yaml
import shutil
import argparse
import os
from collections import defaultdict
import json
import torch


parser = argparse.ArgumentParser(description='PyTorch class Training or Testing')
parser.add_argument('--config_path', default='./experiment/test/config.yaml')
parser.add_argument('--mode', default='test')


def metric_compare(resA,best_res,compare_metric):
    metrics = ['acc','macro_f_score','micro_f_score','kappa']
    compare_metric = set(compare_metric)
    assert compare_metric.issubset(metrics)
    metric_num = len(compare_metric)
    compare_res = 0
    for index in compare_metric:
        if resA[index] > best_res[index]:
            compare_res += 1

    if compare_res / metric_num > 0.5:
        return True
    else:
        return False


def main():
    args = parser.parse_args()
    with open(args.config_path,'rb') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)
    print(config)

    if args.mode == 'train':

        log_dir = os.path.join(os.path.dirname(args.config_path),'logs')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        log_path = os.path.join(log_dir,'log.txt')

        S = classnet_train(config,log_path)    ##train
        S.train()


    elif args.mode == 'test':

        res = test_multiple_models(config,args.config_path)

        print(res)


    elif args.mode == 'trt':

        cls_res,times = run_trt(config,None)
        print("label: ",cls_res)
        print("time consume: ", times)



if __name__=="__main__":
    main()
