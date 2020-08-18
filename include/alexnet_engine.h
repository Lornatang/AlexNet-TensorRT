/*
 * Copyright (c) 2020, Lorna Authors. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ALEXNET_ENGINE_H
#define ALEXNET_ENGINE_H

#include "NvInfer.h"
#include "tensorrt/common.h"
#include "tensorrt/logging.h"
#include "tensorrt/weight.h"

nvinfer1::ICudaEngine *create_alexnet_network(int max_batch_size, nvinfer1::IBuilder *builder,
                                              nvinfer1::DataType data_type, nvinfer1::IBuilderConfig *config,
                                              int number_classes);

void create_alexnet_engine(int max_batch_size, nvinfer1::IHostMemory **model_stream, int number_classes);

#endif// ALEXNET_ENGINE_H