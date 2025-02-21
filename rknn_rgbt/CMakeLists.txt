cmake_minimum_required(VERSION 3.6)

project(rknn_detection_model)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# skip 3rd-party lib dependencies
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# install target and libraries
# CMAKE_SOURCE_DIR: /workspace/chang/rknn_rgbt

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_detection_model_${CMAKE_SYSTEM_NAME})

set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

set(LIB_ARCH aarch64)

include_directories(${CMAKE_SOURCE_DIR})

# rknn api
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/../runtime//${CMAKE_SYSTEM_NAME}/librknn_api)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
set(RKNN_RT_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)

include_directories(${RKNN_API_PATH}/include)
include_directories(${CMAKE_SOURCE_DIR}/../3rdparty)

# opencv

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/../3rdparty/opencv/opencv-linux-aarch64/share/OpenCV)
# 调用 find_package 查找 OpenCV 库，并确保其可用
find_package(OpenCV REQUIRED)

# rga
# comes from https://github.com/airockchip/librga
set(RGA_PATH ${CMAKE_SOURCE_DIR}/../3rdparty/rga/)
set(LIB_ARCH aarch64)
set(RGA_LIB ${RGA_PATH}/libs/Linux//gcc-${LIB_ARCH}/librga.so)
include_directories( ${RGA_PATH}/include)

# mpp
set(MPP_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/mpp)
set(MPP_LIBS ${MPP_PATH}/${CMAKE_SYSTEM_NAME}/${LIB_ARCH}/librockchip_mpp.so)
include_directories(${MPP_PATH}/include)

# zlmediakit
set(ZLMEDIAKIT_PATH ${CMAKE_SOURCE_DIR}/../3rdparty/zlmediakit)
include_directories(${ZLMEDIAKIT_PATH}/include)
set(ZLMEDIAKIT_LIBS ${ZLMEDIAKIT_PATH}/${LIB_ARCH}/libmk_api.so)

if(ZLMEDIAKIT_LIBS)
  add_definitions(-DBUILD_VIDEO_RTSP)
endif()

set(CMAKE_INSTALL_RPATH "lib")

# rknn_yolov5_demo
include_directories(${CMAKE_SOURCE_DIR}/include)

# find_library(STDCXX_FS_LIB stdc++fs /usr/lib/gcc/x86_64-linux-gnu/)

# 定义主可执行文件 rknn_yolov5_demo，并列出所有源文件。
add_executable(rknn_detection_model
  src/main.cc
  src/preprocess.cc
  src/postprocess.cc
)
# 使用 target_link_libraries 将所需的库链接到可执行文件。
target_link_libraries(rknn_detection_model
  ${RKNN_RT_LIB}
  ${RGA_LIB}
  ${OpenCV_LIBS}
)

# target_link_libraries(rknn_detection_model /usr/lib/gcc/x86_64-linux-gnu/7/libstdc++fs.a)

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_detection_model_${CMAKE_SYSTEM_NAME})
install(TARGETS rknn_detection_model DESTINATION ./)

install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
install(PROGRAMS ${RGA_LIB} DESTINATION lib)
install(DIRECTORY model DESTINATION ./model)

file(GLOB IMAGE_FILES "model/*.png")

install(FILES ${IMAGE_FILES} DESTINATION ./model/)

if(ZLMEDIAKIT_LIBS)
  install(PROGRAMS ${ZLMEDIAKIT_LIBS} DESTINATION lib)
endif()
