import torch
import re

def densenet_state_dict(config):
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = torch.load(config.pretrained_path)

    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if key == 'classifier.weight' or key == 'classifier.bias':
            del state_dict[key]
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]

    return state_dict



def model_state_dict(model_dict,config):
    pretrained = torch.load(config.pretrained_path)
    pretrained_dict = {}
    for k in pretrained.keys():
        if k in config.ignore or k not in model_dict:
            continue
        pretrained_dict.update({k: pretrained[k]})
    model_dict.update(pretrained_dict)
    return model_dict