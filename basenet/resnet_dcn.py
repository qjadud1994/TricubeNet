# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Dequan Wang and Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math

import torch
import torch.nn as nn
from basenet.DCNv2.dcn_v2 import DCN
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

BN_MOMENTUM = 0.1

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

            
class Deform_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Deform_conv, self).__init__()
        self.conv = nn.Sequential(
            DCN(in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1, deformable_groups=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    
    
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


                    
class UesNet(nn.Module):

    def __init__(self, block, layers, num_classes, num_layers):
        self.inplanes = 64
        self.deconv_with_bias = False

        super(UesNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if num_layers == 101:
            channels = (2048, 1024, 512, 256)
        else:
            channels = (512, 256, 128, 64)
        
        self.conv1x1_f3 = Deform_conv(channels[1], 256)
        self.conv1x1_f2 = Deform_conv(channels[2], 128)
        self.conv1x1_f1 = Deform_conv(channels[3], 64)
        
        self.deform_conv_f4 = Deform_conv(channels[0], 256)
        self.deform_conv_f3 = Deform_conv(256, 128)
        self.deform_conv_f2 = Deform_conv(128, 64)
        self.deform_conv_f1 = Deform_conv(64, 32)

        self.conv_cls = nn.Sequential(
          nn.Conv2d(32, 256, kernel_size=3, padding=1, bias=False),
          nn.ReLU(inplace=True),
          nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        )
        

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, y):
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.maxpool(y)

        y = self.layer1(y)
        f1 = y
        y = self.layer2(y)
        f2 = y
        y = self.layer3(y)
        f3 = y
        y = self.layer4(y)
        
        """ U network """
        y = self.deform_conv_f4(y) # 256
        y = F.interpolate(y, size=f3.size()[2:], mode='bilinear', align_corners=False)
        y = y + self.conv1x1_f3(f3)
        #y = torch.cat([y, self.conv1x1_f3(f3)], dim=1)
        y = self.deform_conv_f3(y)
        
        y = F.interpolate(y, size=f2.size()[2:], mode='bilinear', align_corners=False)
        #y = torch.cat([y, self.conv1x1_f2(f2)], dim=1)
        y = y + self.conv1x1_f2(f2)
        y = self.deform_conv_f2(y)
        
        y = F.interpolate(y, size=f1.size()[2:], mode='bilinear', align_corners=False)
        #y = torch.cat([y, self.conv1x1_f1(f1)], dim=1)
        y = y + self.conv1x1_f1(f1)
        y = self.deform_conv_f1(y)

        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=False)
        y = self.conv_cls(y)

        return y.permute(0, 2, 3, 1)
        

    def init_weights(self, num_layers):
        
        nn.init.normal_(self.conv_cls[-1].weight.data, mean=0, std=0.001)
        
        url = model_urls['resnet{}'.format(num_layers)]
        pretrained_state_dict = model_zoo.load_url(url)

        self.load_state_dict(pretrained_state_dict, strict=False)
        
        pretrained_state_dict = None

    def head_init(self, ):
        nn.init.normal_(self.conv_cls[-1].weight.data, mean=0, std=0.001)
        
        
        
resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_uesnet101(num_classes, num_layers=101):
    block_class, layers = resnet_spec[num_layers]

    model = UesNet(block_class, layers, num_classes, num_layers)
    model.init_weights(num_layers)
    return model

def get_uesnet18(num_classes, num_layers=18):
    block_class, layers = resnet_spec[num_layers]

    model = UesNet(block_class, layers, num_classes, num_layers)
    model.init_weights(num_layers)
    return model

if __name__ == '__main__':
    
    model = get_pose_net(20).cuda()
    print(model)

    input = torch.randn(1, 3, 256, 256).cuda()

    out = model(input)
    
    print(out.shape)