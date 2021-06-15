**Introduction**

​     该存储库是一个分类网络框架，通过增加具体的网络结构代码，即可进行训练。同时，还包含了`torch->onnx->tensorrt`的部署过程。

**Environment**

​     `torch 1.2.0`

​     `onnx 1.4.0`

​     `tensorrt 5.1.5.0`

​     `cuda 10.1`

​     `cudnn 7.6.5`

​     other dependencies in requirements.txt

**How to use**

​    **train**

​        设置`experiment/test/config.yaml`中的`mode`参数为`train`,以及一些训练相关参数，如`model`，`datasets`等

​              `python main.py --config_path <PATH>`

​       在`deploy`文件夹默认生成`log.txt`，保存了本次训练的loss日志

​    **test**

​        这个过程从训练过程中保存的checkpoint模型中选择在某些指标上表现最好的模型，并可以通过配置`config.yaml`继续将该模型以`pytorch->onnx->tensorrt`的路线进行一步步的转换。

​        设置`experiment/test/config.yaml`中的`mode`参数为`test`,以及配置deploy项确定是否转换到`onnx`及`tensorrt`

​           `python main.py --config_path <PATH>`  

​        在当前路径下默认生成data.txt，描述网络结构和测试结果。

​    **test one model**

​       在`config.yaml`中配置相应的图片文件夹路径

​           `python demo.py  --config_path <PATH> --model_path <MODEL_PATH>`

​    **tensorrt推理**

​       在`config.yaml`中设置`mode`参数为`trt`,并设置`trt`文件的路径（`engine_file_path`）及测试图片的路径（`test_image_path`）

​          `python main.py --config_path <PATH>`                    

​       

​           

​    