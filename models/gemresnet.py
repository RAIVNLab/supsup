# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d

from args import args
from .builder import Builder

def conv3x3(in_planes, out_planes, builder, stride=1):
    return builder.conv3x3(in_planes, out_planes, stride=stride)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, builder, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, builder, stride)
        self.bn1 = builder.batchnorm(planes)
        self.conv2 = conv3x3(planes, planes, builder)
        self.bn2 = builder.batchnorm(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                builder.conv1x1(in_planes, self.expansion * planes, stride=stride),
                builder.batchnorm(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class RN(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf=20):
        super(RN, self).__init__()
        self.in_planes = nf

        builder = Builder()

        self.conv1 = conv3x3(3, nf * 1, builder)
        self.bn1 = builder.batchnorm(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], builder, stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], builder, stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], builder, stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], builder, stride=2)
        self.linear = builder.conv1x1(nf * 8 * block.expansion, num_classes, last_layer=True)

    def _make_layer(self, block, planes, num_blocks, builder, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, builder, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        bsz = x.size(0)
        out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = self.linear(out)
        out = out.view(out.size(0), -1)
        return out


def GEMResNet18():
    return RN(BasicBlock, [2, 2, 2, 2], args.output_size, nf=int(args.width_mult * 20))