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
project(${PROJECT_NAME} VERSION 0.0.1)

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
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Use CUDA
find_package(CUDA REQUIRED)
message(STATUS "CUDA library status:")
message(STATUS "    version: ${CUDA_VERSION}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")

# Add all headers/librarys location to your include paths
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(/usr/local/cuda/lib64)
link_directories(/usr/local/lib/tensorrt)

# Install all library
add_subdirectory(src)

file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/*.cpp)

add_executable(${PROJECT_NAME} ${SOURCES})

# Link all libraries file
target_link_libraries(
        ${PROJECT_NAME}
        ${OpenCV_LIBS}
        nvinfer
        cudart
        tensorrt_common
        tensorrt_logging
        tensorrt_load_weights
        tensorrt_inference
        tensorrt_alexnet_engine)

# install configure
install(
        TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION bin
        PUBLIC_HEADER DESTINATION include/tensorrt)