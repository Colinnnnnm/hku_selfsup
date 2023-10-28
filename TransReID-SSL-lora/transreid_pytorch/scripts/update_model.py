#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import sys
import torch

if __name__ == "__main__":
    input = sys.argv[1]
    #  input = '/mnt1/michuan.lh/log/moco/sysu_200ep/ckpt_0020.pth'
    obj = torch.load(input, map_location="cpu")
    #  obj = obj["state_dict"]
    torch.save(obj,sys.argv[2])
