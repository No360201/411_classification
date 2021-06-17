from test import class_net_test

from easydict import EasyDict
import yaml
import argparse
import os
import torch



parser = argparse.ArgumentParser(description='PyTorch class Testing')
parser.add_argument('--config_path', default='./experiment/test/config.yaml')
parser.add_argument('--model_path', default='/media/a/新加卷/classnet/models/26.pkl')


if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)
    S = class_net_test(config,args.model_path)

    class_idx_dict_cover = S.dataset_train.img_datas.class_to_idx
    class_idx_dict = {k: v for v, k in class_idx_dict_cover.items()}

    for i, data in enumerate(S.data_loader, 0):
        img, label, _ = data
        img = torch.autograd.Variable(img).cuda()
        s, doubt, res = S.test_img(img, score_thr=config.pretrain.con_thr)

        print("img_path: ", _)
        print("prdicted_label: ",class_idx_dict[int(res)])


