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
using namespace std;

static Logger gLogger; /* NOLINT */

static const char *WEIGHTS = "/opt/tensorrt_models/torch/alexnet/alexnet.wts";

// Custom create AlexNet neural network engine
ICudaEngine *create_alexnet_network(int max_batch_size, IBuilder *builder, DataType data_type, IBuilderConfig *config,
                                    int number_classes) {
  INetworkDefinition *model = builder->createNetworkV2(0);

  // Create input tensor of shape {1, 3, 224, 224 } with name INPUT_NAME
  ITensor *data = model->addInput("input", data_type, Dims3{3, 224, 224});
  assert(data);

  std::map<std::string, Weights> weights = load_weights(WEIGHTS);

  // Add convolution layer with 64 outputs and a 11x11 filter.
  IConvolutionLayer *conv1 =
          model->addConvolutionNd(*data, 64, DimsHW{11, 11}, weights["features.0.weight"], weights["features.0.bias"]);
  assert(conv1);
  conv1->setStrideNd(DimsHW{4, 4});
  conv1->setPaddingNd(DimsHW{2, 2});

  // Add activation layer using the ReLU algorithm.
  IActivationLayer *relu1 = model->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
  assert(relu1);

  // Add max pooling layer with stride of 3x3 and kernel size of 2x2.
  IPoolingLayer *pool1 = model->addPoolingNd(*relu1->getOutput(0), PoolingType::kAVERAGE, DimsHW{3, 3});
  assert(pool1);
  pool1->setStrideNd(DimsHW{2, 2});

  // Add convolution layer with 192 outputs and a 5x5 filter.
  IConvolutionLayer *conv2 = model->addConvolutionNd(*pool1->getOutput(0), 192, DimsHW{5, 5},
                                                     weights["features.3.weight"], weights["features.3.bias"]);
  assert(conv2);
  conv2->setPaddingNd(DimsHW{2, 2});

  // Add activation layer using the ReLU algorithm.
  IActivationLayer *relu2 = model->addActivation(*conv2->getOutput(0), ActivationType::kRELU);
  assert(relu2);

  // Add max pooling layer with stride of 3x3 and kernel size of 2x2.
  IPoolingLayer *pool2 = model->addPoolingNd(*relu2->getOutput(0), PoolingType::kAVERAGE, DimsHW{3, 3});
  assert(pool2);
  pool2->setStrideNd(DimsHW{2, 2});

  // Add convolution layer with 384 outputs and a 3x3 filter.
  IConvolutionLayer *conv3 = model->addConvolutionNd(*pool2->getOutput(0), 384, DimsHW{3, 3},
                                                     weights["features.6.weight"], weights["features.6.bias"]);
  assert(conv3);
  conv3->setPaddingNd(DimsHW{1, 1});
  // Add activation layer using the ReLU algorithm.
  IActivationLayer *relu3 = model->addActivation(*conv3->getOutput(0), ActivationType::kRELU);
  assert(relu3);

  // Add convolution layer with 256 outputs and a 3x3 filter.
  IConvolutionLayer *conv4 = model->addConvolutionNd(*relu3->getOutput(0), 256, DimsHW{3, 3},
                                                     weights["features.8.weight"], weights["features.8.bias"]);
  assert(conv4);
  conv4->setPaddingNd(DimsHW{1, 1});
  // Add activation layer using the ReLU algorithm.
  IActivationLayer *relu4 = model->addActivation(*conv4->getOutput(0), ActivationType::kRELU);
  assert(relu4);

  // Add convolution layer with 256 outputs and a 3x3 filter.
  IConvolutionLayer *conv5 = model->addConvolutionNd(*relu4->getOutput(0), 256, DimsHW{3, 3},
                                                     weights["features.10.weight"], weights["features.10.bias"]);
  assert(conv5);
  conv5->setPaddingNd(DimsHW{1, 1});
  // Add activation layer using the ReLU algorithm.
  IActivationLayer *relu5 = model->addActivation(*conv5->getOutput(0), ActivationType::kRELU);
  assert(relu5);

  // Add max pooling layer with stride of 3x3 and kernel size of 2x2.
  IPoolingLayer *pool3 = model->addPoolingNd(*relu5->getOutput(0), PoolingType::kAVERAGE, DimsHW{3, 3});
  assert(pool3);
  pool3->setStrideNd(DimsHW{2, 2});

  // Add fully connected layer with 4096 outputs.
  IFullyConnectedLayer *fc1 = model->addFullyConnected(*pool3->getOutput(0), 4096, weights["classifier.1.weight"],
                                                       weights["classifier.1.bias"]);
  assert(fc1);

  // Add activation layer using the ReLU algorithm.
  IActivationLayer *relu6 = model->addActivation(*fc1->getOutput(0), ActivationType::kRELU);
  assert(relu6);

  IFullyConnectedLayer *fc2 = model->addFullyConnected(*relu6->getOutput(0), 4096, weights["classifier.4.weight"],
                                                       weights["classifier.4.bias"]);
  assert(fc2);

  // Add second fully connected layer with 4096 outputs.
  IActivationLayer *relu7 = model->addActivation(*fc2->getOutput(0), ActivationType::kRELU);
  assert(relu7);

  // Add fully connected layer with number_classes outputs.
  IFullyConnectedLayer *fc3 = model->addFullyConnected(*relu7->getOutput(0), number_classes,
                                                       weights["classifier.6.weight"], weights["classifier.6.bias"]);
  assert(fc3);

  fc3->getOutput(0)->setName("label");
  model->markOutput(*fc3->getOutput(0));

  // Build engine
  builder->setMaxBatchSize(max_batch_size);
  config->setMaxWorkspaceSize(1_GiB);
  config->setFlag(BuilderFlag::kFP16);
  ICudaEngine *engine = builder->buildEngineWithConfig(*model, *config);

  // Don't need the model any more
  model->destroy();

  // Release host memory
  for (auto &memory : weights) free((void *) (memory.second.values));

  return engine;
}

void create_alexnet_engine(int max_batch_size, IHostMemory **model_stream, int number_classes) {
  // Create builder
  report_message(0);
  std::cout << "Creating builder..." << std::endl;
  IBuilder *builder = createInferBuilder(gLogger);
  IBuilderConfig *config = builder->createBuilderConfig();

  // Create model to populate the network, then set the outputs and create an engine
  report_message(0);
  std::cout << "Creating AlexNet network engine..." << std::endl;
  ICudaEngine *engine = create_alexnet_network(max_batch_size, builder, DataType::kFLOAT, config, number_classes);
  assert(engine != nullptr);

  // Serialize the engine
  report_message(0);
  std::cout << "Serialize model engine..." << std::endl;
  (*model_stream) = engine->serialize();
  report_message(0);
  std::cout << "Create AleXNet engine successful." << std::endl;

  // Close everything down
  engine->destroy();
  builder->destroy();
}