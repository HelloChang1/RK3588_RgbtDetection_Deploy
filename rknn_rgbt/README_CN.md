# DETECT MODEL INFERENCE

## 导出rknn模型步骤

请参考 https://github.com/airockchip/rknn_model_zoo/tree/main/models/CV/object_detection/yolo


## 注意事项

1. 使用rknn-toolkit2版本大于等于1.4.0。
2. 切换成自己训练的模型时，请注意对齐anchor等后处理参数，否则会导致后处理解析出错。
3. 官网和rk预训练模型都是检测80类的目标，如果自己训练的模型,需要更改include/postprocess.h中的OBJ_CLASS_NUM以及NMS_THRESH,BOX_THRESH后处理参数。
4. demo需要librga.so的支持,编译使用请参考 https://github.com/airockchip/librga
5. 由于硬件限制，该demo的模型默认把 yolov5 模型的后处理部分，移至cpu实现。本demo附带的模型均使用relu为激活函数，相比silu激活函数精度略微下降，性能大幅上升。

## Aarch64 Linux

### 编译

首先导入GCC_COMPILER，`export GCC_COMPILER=/opt/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu `，然后执行如下命令：

```
./build-linux.sh -t <target> -a <arch> -b <build_type>]

# 例如: 
./build-linux.sh -t rk3588 -a aarch64 -b Release

```

### 推送执行文件到板子


将 install/rknn_yolov5_demo_Linux 拷贝到板子的/userdata/目录.

- 如果使用rockchip的EVB板子，可以使用adb将文件推到板子上：

```
adb push install/rknn_yolov5_demo_Linux /userdata/
```

- 如果使用其他板子，可以使用scp等方式将install/rknn_yolov5_demo_Linux拷贝到板子的/userdata/目录

### 运行

```sh
adb shell
cd /userdata/rknn_yolov5_demo_Linux/

export LD_LIBRARY_PATH=./lib
./rknn_detection_model model/model/rgbt_ca_rtdetrv2_ours_add_original_op19_three_outputs_conv_int8.rknn model/model

```

Note: Try searching the location of librga.so and add it to LD_LIBRARY_PATH if the librga.so is not found on the lib folder.
Using the following commands to add to LD_LIBRARY_PATH.

```sh
export LD_LIBRARY_PATH=./lib:<LOCATION_LIBRGA.SO>
```


### 注意

- 需要根据系统的rga驱动选择正确的librga库，具体依赖请参考： https://github.com/airockchip/librga