#
# Copyright (c) 2020, Lorna Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#

# Check the cmake version number of the system
cmake_minimum_required(VERSION 3.10.2)

# The name of the project
set(PROJECT_NAME alexnet)
project(${PROJECT_NAME} VERSION 1.0)

# Compile using C++17 standard
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -pthread")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "C++17 support has been enabled by default.")

# Set build mode
set(CMAKE_BUILD_TYPE "Release")
# Set install path
set(CMAKE_INSTALL_PREFIX "/usr/local")

# Use OpenCV
find_package(OpenCV REQUIRED)
# Use CUDA
find_package(CUDA REQUIRED)

# Add all headers/librarys location to your include paths
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(/usr/local/cuda/lib64)
link_directories(/usr/local/lib/tensorrt)

# Create all static library
add_library(common common.cpp)
add_library(logging logging.cpp)
add_library(load_weights weight.cpp)
add_library(inference inference.cpp)
add_library(create_alexnet_engine network.cpp)
add_library(serialize_alexnet_engine engine.cpp)
# Link all include file
target_include_directories(common PUBLIC ../include)

# Install  all static library
install(
  TARGETS common
  ARCHIVE DESTINATION lib/tensorrt
  PUBLIC_HEADER DESTINATION include/tensorrt)
install(
  TARGETS logging
  ARCHIVE DESTINATION lib/tensorrt
  PUBLIC_HEADER DESTINATION include/tensorrt)
install(
  TARGETS load_weights
  ARCHIVE DESTINATION lib/tensorrt
  PUBLIC_HEADER DESTINATION include/tensorrt)
install(
  TARGETS inference
  ARCHIVE DESTINATION lib/tensorrt
  PUBLIC_HEADER DESTINATION include/tensorrt)
install(
  TARGETS create_alexnet_engine
  ARCHIVE DESTINATION lib/tensorrt
  PUBLIC_HEADER DESTINATION include/tensorrt)
install(
  TARGETS serialize_alexnet_engine
  ARCHIVE DESTINATION lib/tensorrt
  PUBLIC_HEADER DESTINATION include/tensorrt)
install(FILES ../include/common.h DESTINATION include/tensorrt)
install(FILES ../include/logging.h DESTINATION include/tensorrt)
install(FILES ../include/weight.h DESTINATION include/tensorrt)
install(FILES ../include/inference.h DESTINATION include/tensorrt)
install(FILES ../include/alexnet_network.h DESTINATION include/tensorrt)
install(FILES ../include/alexnet_engine.h DESTINATION include/tensorrt)

# Link common with all libraries
target_link_libraries(common ${OpenCV_LIBS} nvinfer cudart)
target_link_libraries(logging ${OpenCV_LIBS} nvinfer cudart)
target_link_libraries(load_weights ${OpenCV_LIBS} nvinfer cudart logging)
target_link_libraries(inference ${OpenCV_LIBS} nvinfer cudart)
target_link_libraries(create_alexnet_engine ${OpenCV_LIBS} nvinfer cudart
                      logging load_weights)
target_link_libraries(
  serialize_alexnet_engine
  ${OpenCV_LIBS}
  nvinfer
  cudart
  logging
  load_weights
  create_alexnet_engine)
