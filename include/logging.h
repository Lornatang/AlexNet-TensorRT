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

#ifndef LOGGING_H
#define LOGGING_H

#include "NvInfer.h"
#include <ctime>
#include <iostream>

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger {
  public:
  Logger() : Logger(Severity::kWARNING) {}

  explicit Logger(Severity severity) : reportableSeverity(severity) {}

  void log(Severity severity, const char *message) override {
    // suppress messages with severity enum value greater than the reportable
    if (severity > reportableSeverity) return;

    switch (severity) {
      case Severity::kINTERNAL_ERROR:
        std::cerr << "[INTERNAL_ERROR]: ";
        break;
      case Severity::kERROR:
        std::cerr << "[ERROR]: ";
        break;
      case Severity::kWARNING:
        std::cerr << "[WARNING]: ";
        break;
      case Severity::kINFO:
        std::cout << "[INFO]: ";
        break;
      default:
        std::cout << "[INFO]: ";
        break;
    }
    std::cout << message << std::endl;
  }

  Severity reportableSeverity{Severity::kWARNING};
};

void report_message(unsigned int level);

#endif// LOGGING_H