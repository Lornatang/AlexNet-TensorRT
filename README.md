# AlexNet-TensorRT

## Overview

Use the PyTorch framework to build a network model and train the data set, and hand it over to TensorRT for inference.

### Table of contents

- [AlexNet-TensorRT](#alexnet-tensorrt)
  - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [About TensorRT](#about-tensorrt)
    - [About AlexNet-TensorRT](#about-alexnet-tensorrt)
    - [Installation](#installation)
      - [Clone and install requirements](#clone-and-install-requirements)
      - [Download tensorrt weights and tensorrt engine](#download-tensorrt-weights-and-tensorrt-engine)
      - [Download datasets](#download-datasets)
    - [Usage](#usage)
      - [Train](#train)
      - [Inference](#inference)
    - [Credit](#credit)

### About TensorRT

TensorRT is a C++ library for high performance inference on NVIDIA GPUs and deep learning accelerators.
More detail see [https://developer.nvidia.com/tensorrt](https://developer.nvidia.com/tensorrt)

### About AlexNet-TensorRT

This repo uses the TensorRT API to build an engine for a model trained on the [ImageNet2012 dataset](http://www.image-net.org/challenges/LSVRC/2012/).
It creates the network layer by layer, sets up weights and inputs/outputs, and then performs inference.
Both of these samples use the same model weights, handle the same input, and expect similar output.

### Installation

#### Clone and install requirements

```bash
# Install TensorRT-DevelopmentKit
wget https://github.com/Lornatang/TensorRT-Toolkit/archive/TensorRT-ToolKit-v1.1.0.tar.gz
tar xzcf TensorRT-ToolKit-v1.1.0.tar.gz
cd TensorRT-Toolkit-TensorRT-ToolKit-v1.1.0
mkdir build && cd build
cmake ..
sudo make install

# Install this repo
git clone https://github.com/Lornatang/AlexNet-TensorRT.git
cd AlexNet-TensorRT/
pip install -r requirements.txt
```

**In addition, the following conditions should also be met for TensorRT:**

- Cmake >= 3.10.2
- OpenCV >= 4.4.0
- TensorRT >= 7.0

#### Download tensorrt weights and tensorrt engine

```bash
cd weights/
bash download.sh
```

#### Download datasets

See `<AlexNet-TensorRT>/data/README.md`

### Usage

#### Train

```bash
python train.py data
```

#### Inference

1. Compile this sample by running `make` in the `<TensorRT root directory>/build` directory. The binary named `alexnet` will be created in the `<TensorRT root directory>/build/bin` directory.

    ```bash
    cd <TensorRT root directory>
    mkdir build
    cd build
    cmake ..
    make
    ```

    Where `<TensorRT root directory>` is where you clone LeNet-TensorRT.

2. Run the sample to perform inference on the dog:

    ```bash
    ./alexnet --engine <num_classes>// Generate TensorRT inference model.
    ./alexnet --image <image_path> <num_classes> // Reasoning on the picture.
    ```

3. Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following:

    ```text
    # ./alexnet --image ~/Desktop/a.jpg 1000
    [INFO](14:31:3): Read from`/opt/tensorrt_models/torch/alexnet/alexnet.engine` inference engine.
    [WARNING]: Current optimization profile is: 0. Please ensure there are no enqueued operations pending in this context prior to switching profiles
    [INFO](14:31:5): Read image from `/home/unix/Desktop/a.jpg`!
    [INFO](14:31:5): Resize image size to 224 * 224.
    [INFO](14:31:5): Preprocess the input image.
    [INFO](14:31:5): Inference......
    --------                       -----------
    Category                       probability
    --------                       -----------
    tench                          0.515312
    red-breasted merganser          0.42625
    hornbill                       0.334688
    lorikeet                       0.297344
    goldfinch                      0.263594
    ```
  
4. Install into the system directory(optional)

    ```bash
    sudo make install
    # Create lenet engine
    alexnet --engine 1000
    # Test the picture
    alexnet --image ~/Desktop/test.jpg 1000
    ```

### Credit

#### ImageNet Classification with Deep Convolutional Neural Networks

*Alex Krizhevsky,Ilya Sutskever,Geoffrey E. Hinton*

##### Abstract

We trained a large, deep convolutional neural network to classify the 1.2 million
high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 different classes. On the test data, we achieved top-1 and top-5 error rates of 37.5%
and 17.0% which is considerably better than the previous state-of-the-art. The
neural network, which has 60 million parameters and 650,000 neurons, consists
of five convolutional layers, some of which are followed by max-pooling layers,
and three fully-connected layers with a final 1000-way softmax. To make training faster, we used non-saturating neurons and a very efficient GPU implementation of the convolution operation. To reduce overfitting in the fully-connected
layers we employed a recently-developed regularization method called “dropout”
that proved to be very effective. We also entered a variant of this model in the
ILSVRC-2012 competition and achieved a winning top-5 test error rate of 15.3%,
compared to 26.2% achieved by the second-best entry.

[paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

```text
@article{AlexNet,
title:{ImageNet Classification with Deep Convolutional Neural Networks},
author:{Alex Krizhevsky,Ilya Sutskever,Geoffrey E. Hinton},
journal={nips},
year={2012}
}
```
