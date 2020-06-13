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
from alexnet_tensorrt import alexnet


def main():
    model = alexnet()
    model.load_state_dict(torch.load("checkpoints/alexnet.pth")["state_dict"])
    model.to("cuda:0")
    model.eval()

    f = open("checkpoints/alexnet.pb", "w")
    f.write("{}\n".format(len(model.state_dict().keys())))
    for k, v in model.state_dict().items():
        print("key: ", k)
        print("value: ", v.shape)
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")

    print("Model convert successful!")


if __name__ == "__main__":
    main()
