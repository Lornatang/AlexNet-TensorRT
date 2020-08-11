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

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include "NvInfer.h"
#include "common.h"
#include "cuda_runtime_api.h"
#include "opencv2/opencv.hpp"

// stuff we know about the network and the input/output blobs
static const int INPUT_C = 3;
static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int NUMBER_CLASSES = 1000;

const char *INPUT_NAME = "image";
const char *OUTPUT_NAME = "label";

using namespace nvinfer1;

static Logger gLogger;

static bool PairCompare(const std::pair<float, int> &lhs, const std::pair<float, int> &rhs) {
  return lhs.first > rhs.first;
}
// Load weights from files shared with TensorRT samples.
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file) {
  printMessage(0);
  std::cout << "Loading weights from `" << file << "`." << std::endl;
  std::map<std::string, Weights> weightMap;

  // Open weights file
  std::ifstream input(file);
  assert(input.is_open() && "Unable to load weight file.");

  // Read number of weight blobs
  int32_t count;
  input >> count;
  assert(count > 0 && "Invalid weight map file.");

  while (count--) {
    Weights wt{DataType::kFLOAT, nullptr, 0};
    uint32_t size;

    // Read name and type of blob
    std::string name;
    input >> name >> std::dec >> size;
    wt.type = DataType::kFLOAT;

    // Load blob
    uint32_t *val = reinterpret_cast<uint32_t *>(malloc(sizeof(val) * size));
    for (uint32_t x = 0, y = size; x < y; ++x) input >> std::hex >> val[x];

    wt.values = val;
    wt.count = size;
    weightMap[name] = wt;
  }

  return weightMap;
}

// Custom create LeNet neural network engine
ICudaEngine *createEngine(unsigned int maxBatchSize, IBuilder *builder, DataType datatype, IBuilderConfig *config) {
  INetworkDefinition *model = builder->createNetworkV2(0);

  // Create input tensor of shape { 1, 3, 224, 224 } with name INPUT_BLOB_NAME
  ITensor *data = model->addInput(INPUT_NAME, datatype, Dims3{3, INPUT_H, INPUT_W});
  assert(data);

  std::map<std::string, Weights> weights = loadWeights("/opt/tensorrt_models/torch/alexnet/alexnet.wts");

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

  // Add fully connected layer with 1000 outputs.
  IFullyConnectedLayer *fc3 = model->addFullyConnected(*relu7->getOutput(0), NUMBER_CLASSES,
                                                       weights["classifier.6.weight"], weights["classifier.6.bias"]);
  assert(fc3);

  fc3->getOutput(0)->setName(OUTPUT_NAME);
  model->markOutput(*fc3->getOutput(0));

  // Build engine
  builder->setMaxBatchSize(maxBatchSize);
  config->setMaxWorkspaceSize(1_GiB);
  config->setFlag(BuilderFlag::kFP16);
  ICudaEngine *engine = builder->buildEngineWithConfig(*model, *config);

  // Don't need the model any more
  model->destroy();

  // Release host memory
  for (auto &memory : weights) free((void *)(memory.second.values));

  return engine;
}

void serializeEngine(unsigned int maxBatchSize, IHostMemory **modelStream) {
  // Create builder
  IBuilder *builder = createInferBuilder(gLogger);
  IBuilderConfig *config = builder->createBuilderConfig();

  // Create model to populate the network, then set the outputs and create an
  // engine
  printMessage(0);
  std::cout << "Currently creating an inference engine." << std::endl;
  ICudaEngine *engine = createEngine(maxBatchSize, builder, DataType::kFLOAT, config);
  assert(engine != nullptr);

  // Serialize the engine
  (*modelStream) = engine->serialize();

  // Close everything down
  engine->destroy();
  builder->destroy();
}

void inference(IExecutionContext &context, float *input, float *output, int batchSize) {
  const ICudaEngine &engine = context.getEngine();

  // Pointers to input and output device buffers to pass to engine.
  // Engine requires exactly IEngine::getNbBindings() number of buffers.
  assert(engine.getNbBindings() == 2);
  void *buffers[2];

  // In order to bind the buffers, we need to know the names of the input and
  // output tensors. Note that indices are guaranteed to be less than
  // IEngine::getNbBindings()
  const int inputIndex = engine.getBindingIndex(INPUT_NAME);
  const int outputIndex = engine.getBindingIndex(OUTPUT_NAME);

  // Create GPU buffers on device
  CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
  CHECK(cudaMalloc(&buffers[outputIndex], batchSize * NUMBER_CLASSES * sizeof(float)));

  // Create stream
  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));

  // DMA input batch data to device, infer on the batch asynchronously, and DMA
  // output back to host
  CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float),
                        cudaMemcpyHostToDevice, stream));
  context.enqueue(batchSize, buffers, stream, nullptr);
  CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * NUMBER_CLASSES * sizeof(float),
                        cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);

  // Release stream and buffers
  cudaStreamDestroy(stream);
  CHECK(cudaFree(buffers[inputIndex]));
  CHECK(cudaFree(buffers[outputIndex]));
}

void printHelpInfo() {
  printMessage(2);
  std::cerr << "Invalid arguments!" << std::endl;
  std::cout << "Usage: " << std::endl;
  std::cout << "  ./alexnet --engine  // Generate TensorRT inference model." << std::endl;
  std::cout << "  ./alexnet --image ../examples/dog.jpg  // Reasoning on the picture." << std::endl;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    printHelpInfo();
    return -1;
  }

  // create a model using the API directly and serialize it to a stream
  char *trtModelStream{nullptr};
  std::string label_file = "/opt/tensorrt_models/data/class_labels.txt";
  size_t size{0};

  // Subtract mean from image
  float data[3 * INPUT_H * INPUT_W];

  if (std::string(argv[1]) == "--engine") {
    IHostMemory *modelStream{nullptr};
    serializeEngine(1, &modelStream);
    assert(modelStream != nullptr);

    std::ofstream engine("/opt/tensorrt_models/torch/alexnet/alexnet.engine");
    if (!engine) {
      printMessage(2);
      std::cerr << "Could not open plan output file" << std::endl;
      printMessage(0);
      std::cout << "Please refer to the documentation how to generate an inference engine." << std::endl;
      return -1;
    }
    engine.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());

    printMessage(0);
    std::cout << "The inference engine is saved to `/opt/tensorrt_models/torch/alexnet/alexnet.engine`!" << std::endl;

    modelStream->destroy();
    return 1;
  } else if (std::string(argv[1]) == "--image") {
    printMessage(0);
    std::cout << "Read from`/opt/tensorrt_models/torch/alexnet/alexnet.engine` inference engine." << std::endl;
    std::ifstream file("/opt/tensorrt_models/torch/alexnet/alexnet.engine", std::ios::binary);
    if (file.good()) {
      file.seekg(0, file.end);
      size = file.tellg();
      file.seekg(0, file.beg);
      trtModelStream = new char[size];
      assert(trtModelStream);
      file.read(trtModelStream, size);
      file.close();

      printMessage(0);
      std::cout << "Read image from `" << argv[2] << "`!" << std::endl;

      cv::Mat image = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);

      if (image.empty()) {
        printMessage(2);
        std::cerr << "Open image error!" << std::endl;
        return -2;
      }
      printMessage(0);
      std::cout << "Read image successful! " << std::endl;

      printMessage(0);
      std::cout << "Adjust image size to 224 * 224." << std::endl;
      cv::resize(image, image, cv::Size(INPUT_H, INPUT_W));
      printMessage(0);
      std::cout << "Adjust image size successful." << std::endl;

      // Print ASCII representation of digit image
      printMessage(0);
      std::cout << "Input:\n" << std::endl;
      image.convertTo(image, CV_32F);
      image = (image - 127.5) / 128;

      for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
          data[0 * INPUT_H * INPUT_W + i * INPUT_H + j] = static_cast<float>(image.at<cv::Vec3b>(i, j)[0]);
          data[1 * INPUT_H * INPUT_W + i * INPUT_H + j] = static_cast<float>(image.at<cv::Vec3b>(i, j)[1]);
          data[2 * INPUT_H * INPUT_W + i * INPUT_H + j] = static_cast<float>(image.at<cv::Vec3b>(i, j)[2]);
        }
      }
    }
  } else
    return -1;

  IRuntime *runtime = createInferRuntime(gLogger);
  assert(runtime != nullptr);
  ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
  assert(engine != nullptr);
  IExecutionContext *context = engine->createExecutionContext();
  assert(context != nullptr);

  // Run inference
  float prob[NUMBER_CLASSES];
  for (int i = 0; i < 1000; i++) {
    auto start = std::chrono::system_clock::now();
    inference(*context, data, prob, 1);
    auto end = std::chrono::system_clock::now();
  }

  // Destroy the engine
  context->destroy();
  engine->destroy();
  runtime->destroy();

  // Load imagenet labels
  std::vector<std::string> labels_;
  std::ifstream labels(label_file.c_str());
  std::string line;
  while (std::getline(labels, line)) labels_.push_back(std::string(line));

  std::vector<float> probs;

  for (int n = 0; n < NUMBER_CLASSES; n++) probs.push_back(prob[n]);
  std::cout << prob << std::endl;

  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < probs.size(); ++i) pairs.push_back(std::make_pair(probs[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + 5, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < 5; ++i) {
    result.push_back(pairs[i].second);
    std::cout << prob[pairs[i].second] << " - " << labels_[pairs[i].second] << std::endl;
  }
  std::cout << std::endl;

  return 0;
}