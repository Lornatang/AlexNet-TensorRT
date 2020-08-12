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

#include "../include/weight.h"

using namespace nvinfer1;

std::map<std::string, Weights> load_weights(const std::string file) {
  std::map<std::string, Weights> weights;

  report_message(0);
  std::cout << "Opening `" << file << "` TensorRT weights..." << std::endl;
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
    weights[name] = wt;
  }

  return weights;
}