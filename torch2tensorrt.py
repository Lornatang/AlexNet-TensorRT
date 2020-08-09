# Copyright 2020 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import struct

import torch
import torchvision


def create_weights():
    net = torchvision.models.alexnet(pretrained=True).to('cuda:0')
    net.eval()
    torch.save(net, "alexnet.pth")
    del net


def convert_weights():
    net = torch.load('alexnet.pth').to('cuda:0')
    net.eval()

    f = open("alexnet.weights", 'w')
    f.write("{}\n".format(len(net.state_dict().keys())))
    for k, v in net.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")


def main():
    print(f"cuda device count: {torch.cuda.device_count()}")

    create_weights()
    print("Create Model successful!")

    convert_weights()
    print("Model convert successful!")


if __name__ == "__main__":
    main()
