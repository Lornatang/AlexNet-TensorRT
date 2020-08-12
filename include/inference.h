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

#include <assert.h>
#include "NvInfer.h"
#include "common.h"
#include "cuda_runtime_api.h"

// TensorRT general inference function.
void inference(nvinfer1::IExecutionContext &context, float *input, float *output, const char *input_name,
               const char *ouput_name, unsigned int batch_size, unsigned int channel, unsigned int image_height,
               unsigned int image_width, unsigned int number_classes);