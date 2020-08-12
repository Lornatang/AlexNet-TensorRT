#include "../include/inference.h"

using namespace nvinfer1;

void inference(nvinfer1::IExecutionContext &context, float *input,
               float *output, const char *input_name, const char *ouput_name,
               int batch_size, unsigned int channel, unsigned int image_height,
               unsigned int image_width, unsigned int number_classes) {
  const ICudaEngine &engine = context.getEngine();

  // Pointers to input and output device buffers to pass to engine.
  // Engine requires exactly IEngine::getNbBindings() number of buffers.
  assert(engine.getNbBindings() == 2);
  void *buffers[2];

  // In order to bind the buffers, we need to know the names of the input and
  // output tensors. Note that indices are guaranteed to be less than
  // IEngine::getNbBindings()
  const int inputIndex = engine.getBindingIndex(input_name);
  const int outputIndex = engine.getBindingIndex(ouput_name);

  // Create GPU buffers on device
  CHECK(cudaMalloc(&buffers[inputIndex], batch_size * channel * image_height *
                                                 image_width * sizeof(float)));
  CHECK(cudaMalloc(&buffers[outputIndex],
                   batch_size * number_classes * sizeof(float)));

  // Create stream
  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));

  // DMA input batch data to device, infer on the batch asynchronously, and DMA
  // output back to host
  CHECK(cudaMemcpyAsync(buffers[inputIndex], input,
                        batch_size * channel * image_height * image_width *
                                sizeof(float),
                        cudaMemcpyHostToDevice, stream));
  context.enqueue(batch_size, buffers, stream, nullptr);
  CHECK(cudaMemcpyAsync(output, buffers[outputIndex],
                        batch_size * number_classes * sizeof(float),
                        cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);

  // Release stream and buffers
  cudaStreamDestroy(stream);
  CHECK(cudaFree(buffers[inputIndex]));
  CHECK(cudaFree(buffers[outputIndex]));
}