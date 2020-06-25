import math
import torch
import torch.nn as nn
from args import args
from . import modules
from . import init

class Builder(object):
    def __init__(self):
        if args.individual_heads:
            self.last_layer = getattr(modules, 'IndividualHeads')
        self.conv_layer = getattr(modules, args.conv_type)
        self.bn_layer = getattr(modules, args.bn_type)
        self.conv_init = getattr(init, args.conv_init)

    def activation(self):
        return nn.ReLU(inplace=True)

    def conv(
        self,
        kernel_size,
        in_planes,
        out_planes,
        stride=1,
        first_layer=False,
        last_layer=False,
    ):

        if kernel_size == 1:
            if args.individual_heads and last_layer:
                conv = self.last_layer(
                    in_planes, out_planes, kernel_size=1, stride=stride, bias=False
                )
            else:
                conv = self.conv_layer(
                    in_planes, out_planes, kernel_size=1, stride=stride, bias=False
                )
        elif kernel_size == 3:
            conv = self.conv_layer(
                in_planes,
                out_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )
        elif kernel_size == 5:
            conv = self.conv_layer(
                in_planes,
                out_planes,
                kernel_size=5,
                stride=stride,
                padding=2,
                bias=False,
            )
        elif kernel_size == 7:
            conv = self.conv_layer(
                in_planes,
                out_planes,
                kernel_size=7,
                stride=stride,
                padding=3,
                bias=False,
            )
        else:
            return None

        conv.first_layer = first_layer
        conv.last_layer = last_layer
        self.conv_init(conv)
        return conv

    def conv1x1(
        self, in_planes, out_planes, stride=1, first_layer=False, last_layer=False
    ):
        """1x1 convolution with padding"""
        c = self.conv(
            1,
            in_planes,
            out_planes,
            stride=stride,
            first_layer=first_layer,
            last_layer=last_layer,
        )
        return c

    def conv3x3(
        self, in_planes, out_planes, stride=1, first_layer=False, last_layer=False
    ):
        """3x3 convolution with padding"""
        c = self.conv(
            3,
            in_planes,
            out_planes,
            stride=stride,
            first_layer=first_layer,
            last_layer=last_layer,
        )
        return c

    def conv5x5(
        self, in_planes, out_planes, stride=1, first_layer=False, last_layer=False
    ):
        """5x5 convolution with padding"""
        c = self.conv(
            5,
            in_planes,
            out_planes,
            stride=stride,
            first_layer=first_layer,
            last_layer=last_layer,
        )
        return c

    def conv7x7(
        self, in_planes, out_planes, stride=1, first_layer=False, last_layer=False
    ):
        """7x7 convolution with padding"""
        c = self.conv(
            7,
            in_planes,
            out_planes,
            stride=stride,
            first_layer=first_layer,
            last_layer=last_layer,
        )
        return c

    def nopad_conv5x5(
        self, in_planes, out_planes, stride=1, first_layer=False, last_layer=False
    ):

        conv = self.conv_layer(
            in_planes,
            out_planes,
            kernel_size=5,
            stride=stride,
            padding=0,
            bias=False,
        )
        conv.first_layer = first_layer
        conv.last_layer = last_layer
        self.conv_init(conv)
        return conv

    def batchnorm(self, planes):
        return self.bn_layer(planes)