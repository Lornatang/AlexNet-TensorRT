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

#include "../include/alexnet_engine.h"

using namespace nvinfer1;

static Logger gLogger;

void serialize_alexnet_engine(unsigned int max_batch_size, IHostMemory **model_stream, unsigned int number_classes) {
  // Create builder
  report_message(0);
  std::cout << "Creating builder..." << std::endl;
  IBuilder *builder = createInferBuilder(gLogger);
  IBuilderConfig *config = builder->createBuilderConfig();

  // Create model to populate the network, then set the outputs and create an engine
  report_message(0);
  std::cout << "Creating AlexNet network engine..." << std::endl;
  ICudaEngine *engine = create_alexnet_engine(max_batch_size, builder, DataType::kFLOAT, config, number_classes);
  assert(engine != nullptr);

  // Serialize the engine
  report_message(0);
  std::cout << "Serialize model engine..." << std::endl;
  (*model_stream) = engine->serialize();

  // Close everything down
  engine->destroy();
  builder->destroy();
}