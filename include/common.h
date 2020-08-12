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

#ifndef COMMON_H
#define COMMON_H

#include "NvInfer.h"
#include <algorithm>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#define CHECK(status)                                                          \
  do {                                                                         \
    auto ret = (status);                                                       \
    if (ret != 0) {                                                            \
      std::cout << "Cuda failure: " << ret;                                    \
      abort();                                                                 \
    }                                                                          \
  } while (0)

constexpr long double operator"" _GiB(long double val) {
  return val * (1 << 30); /* NOLINT */
}
constexpr long double operator"" _MiB(long double val) {
  return val * (1 << 20); /* NOLINT */
}
constexpr long double operator"" _KiB(long double val) {
  return val * (1 << 10); /* NOLINT */
}

// These is necessary if we want to be able to write 1_GiB instead of 1.0_GiB.
// Since the return type is signed, -1_GiB will work as expected.
constexpr long long int operator"" _GiB(long long unsigned int val) {
  return val * (1 << 30); /* NOLINT */
}
constexpr long long int operator"" _MiB(long long unsigned int val) {
  return val * (1 << 20); /* NOLINT */
}
constexpr long long int operator"" _KiB(long long unsigned int val) {
  return val * (1 << 10); /* NOLINT */
}

// Load MNIST dataset label
std::vector<std::string> load_mnist_labels(const std::string &filename);

// Calculate the probability of the top 5 categories
void output_inference_results(float *prob, std::vector<std::string> labels,
                              int number_classes);

// Similar to the processing of dictionary types in Python
static bool pair_compare(const std::pair<float, int> &lhs,
                         const std::pair<float, int> &rhs) {
  return lhs.first > rhs.first;
}

#endif// COMMON_H