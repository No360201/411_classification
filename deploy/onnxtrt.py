import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
import time
from PIL import Image
import cv2,os
import torchvision
import numpy as np


from .data_preprocessing import get_img_np_nchw
from .common import *


TRT_LOGGER = trt.Logger()

def get_engine(config,calib=None):
    """
    params max_batch_size:      预先指定大小好分配显存
    params onnx_file_path:      onnx文件路径
    params engine_file_path:    待保存的序列化的引擎文件路径
    params fp16_mode:           是否采用FP16
    params save_engine:         是否保存引擎
    returns:                    ICudaEngine
    """
    # 如果已经存在序列化之后的引擎，则直接反序列化得到cudaEngine
    engine_file_path = os.path.join(config.work_dir,'deploy',config.onnx_trt.engine_filename)
    if os.path.exists(engine_file_path):
        print("Reading engine from file: {}".format(engine_file_path))
        with open(engine_file_path, 'rb') as f, \
                trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())  # 反序列化
    else:  # 由onnx创建cudaEngine

        # 使用logger创建一个builder
        # builder创建一个计算图 INetworkDefinition
        # explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        # In TensorRT 7.0, the ONNX parser only supports full-dimensions mode, meaning that your network definition must be created with the explicitBatch flag set. For more information, see Working With Dynamic Shapes.

        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network() as network, \
                trt.OnnxParser(network, TRT_LOGGER) as parser:  # 使用onnx的解析器绑定计算图，后续将通过解析填充计算图
            builder.max_workspace_size = 1 << 30  # 预先分配的工作空间大小,即ICudaEngine执行时GPU最大需要的空间
            builder.max_batch_size = config.onnx_trt.max_batch_size  # 执行时最大可以使用的batchsize

            if config.onnx_trt.fp16_mode:
                assert (builder.platform_has_fast_fp16 == True), "not support fp16"
                builder.fp16_mode = True

            elif config.onnx_trt.int8_mode:
                assert (builder.platform_has_fast_int8 == True), "not support int8"
                builder.int8_mode = True
                builder.int8_calibrator = calib

            # 解析onnx文件，填充计算图
            onnx_file_path = os.path.join(config.work_dir,'deploy',config.onnx_trt.onnx_filename)
            if not os.path.exists(onnx_file_path):
                quit("ONNX file {} not found!".format(onnx_file_path))
            print('loading onnx file from path {} ...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:  # 二值化的网络结果和参数
                print("Begining onnx file parsing")
                parser.parse(model.read())  # 解析onnx文件
            # parser.parse_from_file(onnx_file_path) # parser还有一个从文件解析onnx的方法

            print("Completed parsing of onnx file")
            # 填充计算图完成后，则使用builder从计算图中创建CudaEngine
            print("Building an engine from file {}' this may take a while...".format(onnx_file_path))

            print(network.get_layer(network.num_layers - 1).get_output(0).shape)

            engine = builder.build_cuda_engine(network)  # 注意，这里的network是INetworkDefinition类型，即填充后的计算图
            print("Completed creating Engine")
            if config.onnx_trt.save_engine:  # 保存engine供以后直接反序列化使用
                with open(engine_file_path, 'wb') as f:
                    f.write(engine.serialize())  # 序列化
            return engine


def postprocess_the_outputs(h_outputs, shape_of_output):
    h_outputs = h_outputs.reshape(*shape_of_output)
    return h_outputs


def run_trt(config,calib):
    image = get_img_np_nchw(config.onnx_trt.test_image_path ,config.onnx_trt.max_batch_size).astype(np.float32)
    with get_engine(config,calib) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        # Do inference
        print('Running inference on image {}...'.format(config.onnx_trt.test_image_path))
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        inputs[0].host = image.reshape(-1)

        t1 = time.time()
        trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        t2 = time.time()

    shape_of_output = (config.onnx_trt.max_batch_size,config.model.kwargs.num_classes)
    result = postprocess_the_outputs(trt_outputs[0], shape_of_output)

    print('result',np.where(np.max(result)))
    print("Inference time with TensorRT time ",t2 - t1)

    return result,t2 - t1









