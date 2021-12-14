import torch
import torchvision
import torch.onnx
import onnx
import os
from core.class_net_test import class_net_test

class torch2onnx(class_net_test):

    def __init__(self,config,best_model_path):
        class_net_test.__init__(self,config,best_model_path)
        self.batch_size = 1
        print(self.model)

    def transfer(self,config):
        if config.torch2onnx.dynamic_axis:
            self.batch_size = config.torch2onnx.batch_size

        input_w,input_h = config.datasets.transform.test.kwargs.size
        input_c = 3

        # print(input_w,input_h)

        input = torch.randn(self.batch_size,input_c,input_h,input_w).cuda()

        input_name = ['input']
        output_name = ['output']

        save_dir = config.work_dir + "deploy/"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = save_dir + config.torch2onnx.save_filename

        torch.onnx.export(self.model, input,
                          save_path, verbose=False,
                          input_names=input_name, output_names=output_name,
                          dynamic_axes = {'input': {0: 'batch_size'},
                                          'output': {0: 'batch_size'}})

        # onnx_model = onnx.load(save_path)
        # onnx.checker.check_model(onnx_model)




