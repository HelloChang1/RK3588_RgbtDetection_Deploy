#!/bin/bash

GCC_COMPILER=/opt/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu 

set -e
# 脚本在任意命令返回非零状态（即发生错误）时立即退出
echo "$0 $@"
# 输出脚本的名称（$0）以及传给它的所有参数（$@）
while getopts ":t:a:b" opt; do
# getopts 函数来解析命令行选项。其中 : 后面的字符表明该选项需要参数（t 和 a），b 则不需要参数
  # 使用 case 语句来处理不同的选项
  case $opt in
    # 提供了 -t 选项，则将其参数（目标 SoC）保存在变量 TARGET_SOC 中
    t)
      TARGET_SOC=$OPTARG
      ;;
    a)
      TARGET_ARCH=$OPTARG
      ;;
    b)
      BUILD_TYPE=$OPTARG
      ;;
    :)
      echo "Option -$OPTARG requires an argument." 
      exit 1
      ;;
    ?)
      echo "Invalid option: -$OPTARG index:$OPTIND"
      ;;
  esac
done

# 这行检查 TARGET_SOC 是否为空。如果为空，则输出用法说明并退出。
if [ -z ${TARGET_SOC} ];then
  echo "$0 -t <target> -a <arch> [-b <build_type>]"
  echo ""
  echo "    -t : target (rk3566/rk3568/rk3562/rk3576/rk3588)"
  echo "    -a : arch (aarch64/armhf)"
  echo "    -b : build_type(Debug/Release)"
  echo "such as: $0 -t rk3588 -a aarch64 -b Release"
  echo ""
  exit -1
fi

# 检查是否设置了 GCC_COMPILER 环境变量。如果没有设置，则输出错误信息并退出
if [[ -z ${GCC_COMPILER} ]];then
  echo "Please set GCC_COMPILER for $TARGET_SOC"
  echo "such as export GCC_COMPILER=~/opt/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu"
  exit
fi

echo "$GCC_COMPILER"
export CC=${GCC_COMPILER}-gcc
export CXX=${GCC_COMPILER}-g++
# 设置 C 和 C++ 编译器变量 CC 和 CXX。

# 检查 CC 变量中指定的编译器命令是否存在。如果命令不存在，则输出错误信息。
if command -v ${CC} >/dev/null 2>&1; then
    :
else
    echo "${CC} is not available"
    echo "Please set GCC_COMPILER for $TARGET_SOC"
    echo "such as export GCC_COMPILER=~/opt/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu"
    exit
fi

# 设置构建类型
# Debug / Release
if [[ -z ${BUILD_TYPE} ]];then
    BUILD_TYPE=Release
fi

# 使用 case 语句将特定的目标 SoC 映射到相应的值。
case ${TARGET_SOC} in
    rk3588)
        TARGET_SOC="RK3588"
        ;;
    *)
        echo "Invalid target: ${TARGET_SOC}"
        echo "Valid target: rk3562,rk3566,rk3568,rk3576,rk3588"
        exit -1
        ;;
esac

# 定义目标平台和构建目录
TARGET_PLATFORM=${TARGET_SOC}_linux

if [[ -n ${TARGET_ARCH} ]];then
TARGET_PLATFORM=${TARGET_PLATFORM}_${TARGET_ARCH}
fi

# 获取脚本的根目录路径。
# ROOT_PWD: /workspace/chang/rknn_rgbt/
ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )
# 根据目标平台和构建类型定义构建目录。
BUILD_DIR=${ROOT_PWD}/build/build_${TARGET_PLATFORM}_${BUILD_TYPE}


echo "==================================="
echo "TARGET_SOC=${TARGET_SOC}"
echo "TARGET_ARCH=${TARGET_ARCH}"
echo "BUILD_TYPE=${BUILD_TYPE}"
echo "BUILD_DIR=${BUILD_DIR}"
echo "CC=${CC}"
echo "CXX=${CXX}"
echo "==================================="

# 检查构建目录是否存在，若不存在则创建该目录。
if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p ${BUILD_DIR}
fi

#  从上级两级目录配置CMake
# cmake ../.. :  /workspace/chang/rknn_rgbt/
cd ${BUILD_DIR}
cmake ../.. \
    -DTARGET_SOC=${TARGET_SOC} \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=${TARGET_ARCH} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_C_COMPILER=${CC} \
    -DCMAKE_CXX_COMPILER=${CXX}
make -j4
make install