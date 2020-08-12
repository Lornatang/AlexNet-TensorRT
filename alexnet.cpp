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

#include "include/alexnet_engine.h"
#include "include/alexnet_network.h"
#include "include/common.h"
#include "include/inference.h"
#include "include/logging.h"
#include "include/weight.h"
#include "opencv2/opencv.hpp"
#include <cstdlib>
#include <unistd.h>

// stuff we know about the network and the input/output blobs
static const unsigned int BATCH_SIZE = 1;
static const unsigned int INPUT_C = 3;
static const unsigned int INPUT_H = 224;
static const unsigned int INPUT_W = 224;

const char *INPUT_NAME = "input";
const char *OUTPUT_NAME = "label";
const char *LABEL_FILE = "/opt/tensorrt_models/data/imagenet1000.txt";
const char *ENGINE_FILE = "/opt/tensorrt_models/torch/alexnet/alexnet.engine";

using namespace nvinfer1;

static Logger gLogger; /* NOLINT */

int main(int argc, char **argv) {
  if (argc < 2) {
    report_message(2);
    std::cerr << "Invalid arguments!" << std::endl;
    std::cout << "Usage: " << std::endl;
    std::cout << "  alexnet --engine <num_classes>// Generate TensorRT "
                 "inference model."
              << std::endl;
    std::cout << "  alexnet --image <image_path> <num_classes> // Reasoning on "
                 "the picture."
              << std::endl;
    return -1;
  }

  // create a model using the API directly and serialize it to a stream
  char *trtModelStream{nullptr};
  size_t size{0};

  if (std::string(argv[1]) == "--engine") {
    IHostMemory *model_stream{nullptr};
    report_message(0);
    std::cout << "Start serialize AlexNet network engine." << std::endl;
    serialize_alexnet_engine(1, &model_stream, atoi(argv[2]));
    assert(model_stream != nullptr);

    std::ofstream engine(ENGINE_FILE);
    if (!engine) {
      report_message(2);
      std::cerr << "Could not open plan output file" << std::endl;
      report_message(0);
      std::cout << "Please refer to the documentation how to generate an "
                   "inference engine."
                << std::endl;
      return -1;
    }
    engine.write(reinterpret_cast<const char *>(model_stream->data()),
                 model_stream->size());

    report_message(0);
    std::cout << "The inference engine is saved to `" << ENGINE_FILE << "`!"
              << std::endl;

    model_stream->destroy();
    return 1;
  } else if (std::string(argv[1]) == "--image") {
    report_message(0);
    std::cout << "Read from`" << ENGINE_FILE << "` inference engine."
              << std::endl;
    std::ifstream file(ENGINE_FILE, std::ios::binary);
    if (file.good()) {
      file.seekg(0, std::ifstream::end);
      size = file.tellg();
      file.seekg(0, std::ifstream::beg);
      trtModelStream = new char[size];
      assert(trtModelStream);
      file.read(trtModelStream, size);
      file.close();
    }
  } else
    return -1;

  // Get abs file
  char *abs_path, *filename;
  filename = realpath(argv[2], abs_path);
  assert(filename != nullptr && "File does not exist");

  // Get Model ouput size
  int number_classes = atoi(argv[3]);

  IRuntime *runtime = createInferRuntime(gLogger);
  assert(runtime != nullptr);
  ICudaEngine *engine =
          runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
  assert(engine != nullptr);
  IExecutionContext *context = engine->createExecutionContext();
  assert(context != nullptr);

  // Read a digit file
  float data[INPUT_C * INPUT_H * INPUT_W];
  cv::Mat raw_image, image;

  report_message(0);
  std::cout << "Read image from `" << filename << "`!" << std::endl;

  raw_image = cv::imread(filename);
  if (raw_image.empty()) {
    report_message(2);
    std::cerr << "Open image error!" << std::endl;
    return -2;
  }

  report_message(0);
  std::cout << "Resize image size to 224 * 224." << std::endl;
  cv::resize(raw_image, image, cv::Size(INPUT_H, INPUT_W));

  report_message(0);
  std::cout << "Preprocess the input image." << std::endl;
  image.convertTo(image, CV_32F);
  image = (image - 127.5) / 128;
  for (int i = 0; i < image.rows; ++i) {
    for (int j = 0; j < image.cols; ++j) {
      data[0 * INPUT_H * INPUT_W + i * INPUT_H + j] =
              static_cast<float>(image.at<cv::Vec3b>(i, j)[0]);
      data[1 * INPUT_H * INPUT_W + i * INPUT_H + j] =
              static_cast<float>(image.at<cv::Vec3b>(i, j)[1]);
      data[2 * INPUT_H * INPUT_W + i * INPUT_H + j] =
              static_cast<float>(image.at<cv::Vec3b>(i, j)[2]);
    }
  }

  // Run inference
  float prob[number_classes];

  report_message(0);
  std::cout << "Inference......" << std::endl;
  for (int i = 0; i < 1000; i++) {
    auto start = std::chrono::system_clock::now();
    inference(*context, data, prob, INPUT_NAME, OUTPUT_NAME, BATCH_SIZE,
              INPUT_C, INPUT_H, INPUT_W, number_classes);
    auto end = std::chrono::system_clock::now();
  }

  // Destroy the engine
  context->destroy();
  engine->destroy();
  runtime->destroy();

  // Load dataset labels
  std::vector<std::string> labels = load_mnist_labels(LABEL_FILE);

  // Formatted output object probability
  output_inference_results(prob, labels, number_classes);
  std::cout << std::endl;

  return 0;
}