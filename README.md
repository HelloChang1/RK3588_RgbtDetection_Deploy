# 目录结构
```
├── 3rdparty     // 用于存放编译的第三方库，包括opencv等
├── rknn_rhbt    // 用于存放模型转换与推理文件
|—————convert    // 模型转换与量化文件
|—————src        // c++推理文件
├── runtime      // 用于存放板端so文件                                  
```
# DETECT MODEL INFERENCE(./rknn_rgbt)

## 模型转换：导出rknn模型步骤
详见/workspace/chang/rknn_rgbt/convert/
可参考 https://github.com/airockchip/rknn_model_zoo/tree/main/models/CV/object_detection/yolo


## 注意事项

1. 使用rknn-toolkit2版本大于等于1.4.0。
2. 切换成自己训练的模型时，请注意对齐anchor等后处理参数，否则会导致后处理解析出错。
3. 本项目模型检测6类的目标，如果自己训练的模型,需要更改include/postprocess.h中的OBJ_CLASS_NUM以及NMS_THRESH,BOX_THRESH后处理参数。
4. 如果需要librga.so的支持,编译使用请参考 https://github.com/airockchip/librga
5. 由于硬件限制，模型把后处理部分，移至cpu实现。

## 模型推理（Aarch64 Linux 环境）

### 交叉编译

具体详见：rknn_rgbt/build-linux.sh

- 首先导入GCC_COMPILER，`export GCC_COMPILER=/opt/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu `，然后执行如下命令：

- 然后执行交叉编译命令
```
./build-linux.sh -t <target> -a <arch> -b <build_type>]

# 例如: 
./build-linux.sh -t rk3588 -a aarch64 -b Release

```

### 推送执行文件到板子

具体详见：/workspace/chang/rknn_rgbt/push_3588.sh

- 将 install/rknn_detection_model_Linux拷贝到板子的 拷贝到板子的指定目录.

- 使用scp等方式将install/rknn_detection_model_Linux拷贝到板子的指定目录

**以上步骤均在rknn_toolkit2的虚拟环境中进行**

**以下步骤均在RK3588板端中进行**

### 板端运行

```sh
cd /{your path}/rknn_yolov5_demo_Linux/

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