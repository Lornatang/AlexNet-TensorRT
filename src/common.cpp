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

#include "../include/common.h"

using namespace std;

vector<string> load_mnist_labels(const string &filename) {
  vector<string> labels;
  ifstream labels_file(filename.c_str());
  string line;
  while (getline(labels_file, line)) labels.emplace_back(line);

  return labels;
}

void output_inference_results(float *prob, vector<string> labels,
                              int number_classes) {
  // Calculate the probability of the top 5 categories
  std::vector<float> probs;
  std::vector<std::pair<float, int>> pairs;
  std::vector<int> results;

  probs.reserve(number_classes);
  for (int n = 0; n < number_classes; n++) { probs.push_back(prob[n]); }

  // Sort the categories in the array
  pairs.reserve(probs.size());
  for (size_t i = 0; i < probs.size(); ++i) { pairs.emplace_back(probs[i], i); }
  std::partial_sort(pairs.begin(), pairs.begin() + 5, pairs.end(),
                    pair_compare);

  // Formatted output and display
  std::cout << std::left << std::setw(30) << "--------" << std::right
            << std::setw(12) << "-----------" << std::endl;
  std::cout << std::left << std::setw(30) << "Category" << std::right
            << std::setw(12) << "probability" << std::endl;
  std::cout << std::left << std::setw(30) << "--------" << std::right
            << std::setw(12) << "-----------" << std::endl;
  for (int i = 0; i < 5; ++i) {
    results.push_back(pairs[i].second);
    std::cout << std::left << std::setw(30) << labels[pairs[i].second]
              << std::right << std::setw(9) << prob[pairs[i].second] / 100
              << std::endl;
  }
}
