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

#include "../include/logging.h"

void report_message(unsigned int level) {
  std::time_t now = time(0);
  std::tm* ltm = localtime(&now);
  switch (level) {
    case 0:
      std::cout << "[INFO]"
                << "(" << ltm->tm_hour << ":" << ltm->tm_min << ":" << ltm->tm_sec << "): ";
      break;
    case 1:
      std::cout << "[WARNING]"
                << "(" << ltm->tm_hour << ":" << ltm->tm_min << ":" << ltm->tm_sec << "): ";
      break;
    case 2:
      std::cerr << "[ERROR]"
                << "(" << ltm->tm_hour << ":" << ltm->tm_min << ":" << ltm->tm_sec << "): ";
      break;
    default:
      std::cout << "[INFO]"
                << "(" << ltm->tm_hour << ":" << ltm->tm_min << ":" << ltm->tm_sec << "): ";
      break;
  }
}