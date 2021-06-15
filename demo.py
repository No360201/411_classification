from test import class_net_test

from easydict import EasyDict
import yaml
import argparse



parser = argparse.ArgumentParser(description='PyTorch class Testing')
parser.add_argument('--config_path', default='./experiment/test/config.yaml')
parser.add_argument('--model_path', default='/media/a/新加卷/classnet/models/26.pkl')


if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)
    S = class_net_test(config,args.model_path)
    S.test(config)

