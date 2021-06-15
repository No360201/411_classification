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

from model import build_model
from datasets import build_dataset,build_aug
from utils.optimizer import build_optimizer
from utils.scheduler import get_scheduler
from utils.criterion import build_criterion



class class_net_test(object):
    def __init__(self,config,model_path):
        self.config=config
        self.creat_dataset(self.config.datasets)        
        self.creat_model(self.config.model)
        self.load_pretrained(self.config,model_path)

    def creat_model(self,config):
        print("creating model")
        self.model=build_model(config)
        if torch.cuda.is_available():
            self.model.cuda()

    def creat_dataset(self,config):
        print("creating data")
        self.transform_train,self.transform_test=build_aug(config.transform)
        self.dataset_train,self.dataset_test=build_dataset(config,self.transform_train,self.transform_test)
        self.data_loader=DataLoader(self.dataset_test, batch_size=1, shuffle=False)

    def load_pretrained(self,config,model_path):
        print("loading")
        config=config.pretrain
        if config.load_all_model:
            dic=torch.load(model_path,map_location=lambda storage, loc: storage)
            self.model.load_state_dict(dic)
        else:
            model_dict = self.model.state_dict()
            pretrained = torch.load(model_path)
            pretrained_dict = {}
            for k in pretrained.keys():
                if k in config.ignore:
                    continue
                pretrained_dict.update({k: pretrained[k]})
            model_dict.update(pretrained_dict) 
            self.model.load_state_dict(model_dict)
        self.model.cuda()
        self.model.eval()

    def test_img(self,img,score_thr):
        # img_test = torch.unsqueeze(img, 0)
        output = self.model(img)

        output_max = torch.max(torch.softmax(output,1),1)
        comp = output_max[0] < score_thr
        doubt = comp.cpu().numpy()

        _,predicted = torch.max(output.data,1)

        all_top3 = []

        for idx in range(output.size(0)):
            output3 = output[idx].cpu().data.numpy()
            top3=heapq.nlargest(3,range(len(output3)),output3.take)
            all_top3.append(top3)

        return all_top3,doubt,predicted


    def calc_acc(self,labels,pred_labels):

        total_val = labels.shape[0]
        correct_val = np.sum(pred_labels == labels)
        acc = correct_val / total_val
        return  acc


    def calc_class_fp(self,labels,pred_labels,cls):
        ###just test for two classes:2and 268

        # cls = ['266','268']
        cls_fp = [[] for i in range(len(cls))]
        for idx in range(labels.shape[0]):
            if self.class_idx_dict[labels[idx]] in cls:
                if pred_labels[idx] != labels[idx]:
                    cls_fp[cls.index(self.class_idx_dict[labels[idx]])].append(self.class_idx_dict[pred_labels[idx]])
        return cls_fp

    def calc_category_tp(self,labels,pred_labels):
        label_gt = np.array([0] * len(self.class_idx_dict))
        label_pred = np.array([0] * len(self.class_idx_dict))
        label_tp = np.array([0] * len(self.class_idx_dict))

        for l in labels:
            label_gt[l] += 1
        for l in pred_labels:
            label_pred[l] += 1
        for l in pred_labels[pred_labels == labels]:
            label_tp[l] += 1
        print(label_gt)
        print(label_tp)
        print(label_pred)


        return  label_gt,label_pred,label_tp

    def calc_prec_rec(self,labels,pred_labels):
        label_gt,label_pred,label_tp = self.calc_category_tp(labels,pred_labels)
        prec = np.around(label_tp / label_pred, decimals=4)
        rec = np.around(label_tp / label_gt, decimals=4)


        index = [i for i in range(len(self.class_idx_dict))]
        class_index = list(map(self.class_idx_dict.get,index))

        print("index",index)
        print("class_index",class_index)

        score_thresh = np.arange(0, 1.05, 0.05)
        prec_thresh_num = np.array([0] * len(score_thresh))
        for i in range(1, len(score_thresh)):
            res = prec <= score_thresh[i]
            print("res",res)

            temp = np.sum(res) - np.sum(prec_thresh_num[:i])
            prec_thresh_num[i - 1] += temp

        prec_dict = dict(zip(class_index, prec))
        rec_dict = dict(zip(class_index, rec))

        prec_sort = dict(sorted(prec_dict.items(), key=lambda x: x[1]))
        rec_sort = dict(sorted(rec_dict.items(), key=lambda x: x[1]))


        return prec_sort,rec_sort,prec_thresh_num

    def calc_doubt_ratio(self,labels,doubt_labels):
        total_val = labels.shape[0]
        doubt_val = np.sum(doubt_labels)
        doubt_ratio = doubt_val / total_val

        return doubt_ratio

    def calc_F_score(self,labels,pred_labels,a):
        label_gt,label_pred,label_tp = self.calc_category_tp(labels,pred_labels)
        prec = np.around(label_tp / label_pred, decimals=4)
        rec = np.around(label_tp / label_gt, decimals=4)

        total_score = 0
        for idx in range(len(prec)):
            total_score += ((1 + a ** 2) * prec[idx] * rec[idx]) / ((a ** 2) * (prec[idx] + rec[idx]))
        macro_f_score = total_score / len(prec)

        prec_total = np.sum(label_tp) / np.sum(label_pred)
        rec_total = np.sum(label_tp) / np.sum(label_gt)
        micro_f_score = ((1 + a ** 2) * prec_total * rec_total) / ((a ** 2) * (prec_total + rec_total))

        return macro_f_score,micro_f_score


    def calc_kappa(self,labels,pred_labels):
        label_gt,label_pred,label_tp = self.calc_category_tp(labels,pred_labels)
        P0 = np.sum(label_tp) / np.sum(label_pred)
        Pe = 0
        for l in range(len(label_pred)):
            Pe += (label_gt[l] * label_pred[l]) / (np.sum(label_gt) ** 2)
        K = (P0 - Pe) / (1 - Pe)

        return K




    def test(self,config):
        self.class_idx_dict_cover = self.dataset_test.img_val.class_to_idx
        self.class_idx_dict = {k:v for v,k in self.class_idx_dict_cover.items()}

        # print(self.class_idx_dict)

        labels = []
        pred_labels = []
        doubt_labels = []
        time_consume = 0

        for i, data in enumerate(self.data_loader, 0):
            img,label,_=data
            img = torch.autograd.Variable(img).cuda()
            label = torch.autograd.Variable(label).cuda()


            start = time.time()
            s, doubt,res = self.test_img(img,score_thr=self.config.pretrain.con_thr)
            end = time.time()
            time_consume = (end - start) / self.data_loader.batch_size


            for l in label:
                labels.append(l.cpu().numpy())
            for pred_l in res:
                pred_labels.append(pred_l.cpu().numpy())
            for d in doubt:
                doubt_labels.append(d)

            for i in range(len(_)):
                top3_i = []
                for temp in s[i]:
                    top3_i.append(self.class_idx_dict[temp])
                # print("image: ",_[i].split("/")[-1]," if doubt: ",doubt[i], " top-3: ",top3_i)
                print("image: ",_[i]," if doubt: ",doubt[i], " top-3: ",top3_i)



        labels = np.array(labels)
        pred_labels = np.array(pred_labels)
        doubt_labels = np.array(doubt_labels)
        
        # cls = ['266','268']
        # num_fp = self.calc_class_fp(labels,pred_labels,cls)
        # _gt,_,_tp = self.calc_category_tp(labels,pred_labels)
        # for _c in cls:
        #     print(_c,num_fp[cls.index(_c)],_gt[self.class_idx_dict_cover[_c]],_tp[self.class_idx_dict_cover[_c]])


        metric_dict = dict()

        if config.evaluation.accuracy:
            acc = self.calc_acc(labels,pred_labels)

            metric_dict['acc'] = acc

            print("acc: ", acc)

        if config.evaluation.category_prec_rec:
            prec_sort,rec_sort,prec_thresh_num = self.calc_prec_rec(labels,pred_labels)

            metric_dict['prec_class'] = prec_sort
            metric_dict['rec_class'] = rec_sort
            metric_dict['prec_thresh_num'] = prec_thresh_num[:-1].tolist()

            print("prec: ", prec_sort)
            print("rec: ", rec_sort)
            print("prec_thresh_num: ", prec_thresh_num[:-1])


        if config.evaluation.doubt_ratio:
            doubt_ratio = self.calc_doubt_ratio(labels,doubt_labels)

            metric_dict['doubt_ratio'] = doubt_ratio

            print("doubt_ratio: ", doubt_ratio)


        if config.evaluation.F_score > 0:
            a = config.evaluation.F_score
            macro_f_score,micro_f_score = self.calc_F_score(labels,pred_labels,a)

            metric_dict['macro_f_score'] = macro_f_score
            metric_dict['micro_f_score'] = micro_f_score

            print("a = ", a ," macro_f_score: ",macro_f_score)
            print("a = ", a ," micro_f_score: ",micro_f_score)


        if config.evaluation.Kappa:
            K = self.calc_kappa(labels,pred_labels)

            metric_dict['kappa'] = K

            print("kappa: ", K)

        return metric_dict,time_consume


