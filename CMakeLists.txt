cmake_minimum_required(VERSION 3.10)
project(AlexNet_TensorRT)

add_definitions(-std=c++14)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/usr/local/cuda-10.2/targets/x86_64-linux/include)
link_directories(/usr/local/cuda-10.2/targets/x86_64-linux/lib)

add_executable(AlexNet_TensorRT
        alexnet.cpp)
target_link_libraries(AlexNet_TensorRT nvinfer)
target_link_libraries(AlexNet_TensorRT cudart)

add_definitions(-O2 -pthread)

