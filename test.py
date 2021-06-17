# -*- coding: utf-8 -*-
import sys
import torch
import os
import cv2
from torchvision import transforms
import time
import numpy as np
import heapq
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

from solver.class_net_test import class_net_test

from deploy.torch2onnx import torch2onnx
from deploy.calib import CenterNetEntropyCalibrator
from deploy.onnxtrt import get_engine,run_trt



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


def test_multiple_models(config,config_path):
    info_init = 1

    metric_info = defaultdict(dict)
    best_model_metric = defaultdict(list)
    best_model_path = ''

    compare_metric = config.evaluation.compare_metric

    print(os.listdir(config.pretrain.checkpoint_path))
    for model in os.listdir(config.pretrain.checkpoint_path):
        print("aaaa")
        print(model)
        if model.endswith(".pkl"):
            model_path = config.pretrain.checkpoint_path + model
            print(model_path)
            S = class_net_test(config,model_path)
            res, time_consume = S.test(config)

            if not best_model_metric:
                best_model_metric = res
                best_model_path = model_path

            compare_flag = metric_compare(res,best_model_metric,compare_metric)

            if compare_flag:
                best_model_path = model_path

            if info_init:
                metric_info['network'] = S.model
                metric_info['checkpints_metric'] = list()
                metric_info['best_checkpoint'] = dict()
                info_init = 0

            checkpoint_info = dict()
            checkpoint_info['checkpoint_path'] = model_path
            checkpoint_info['measurement_result'] = res
            metric_info['checkpints_metric'].append(checkpoint_info)

    best_checkpoint_info = dict()
    best_checkpoint_info['checkpoint_path'] = best_model_path
    best_checkpoint_info['time_consume_torch'] = time_consume
    metric_info['best_checkpoint'] = best_checkpoint_info

    if hasattr(torch.cuda,'empty_cache'):
        torch.cuda.empty_cache()


    if config.deploy.onnx:
        S = torch2onnx(config,best_model_path)         ##torch_onnx
        S.transfer(config)

        if config.deploy.tensorrt:
            calib = None
                               ##onnx_trt
            if config.onnx_trt.int8_mode:
                cache_path = os.path.join(config.work_dir,'deploy',config.onnx_trt.cache_filename)
                calib = CenterNetEntropyCalibrator(config.onnx_trt.calib_images_dir,cache_path)
                get_engine(config,calib)         ## 生成engine
                cls_res, times = run_trt(config,calib)            ## 运行engine

                metric_info['best_checkpoint']['time_consume_tensorrt'] = times

    test_data_path = os.path.join(os.path.dirname(config_path),'models','test_results.txt')

    with open(test_data_path, 'w') as f:
        for key,val in metric_info.items():
            if isinstance(val,list):
                f.write(str(key) + " :" + "\n" )
                f.write("[" + "\n")
                for item in val:
                    f.write(str(item) + " : " + "\n")
                f.write("]" + "\n")
            else:
                f.write(str(key) + " : " + str(val) + "\n")

    f.close()

    return best_checkpoint_info






