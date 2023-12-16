"""
EE 5561 - Image Processing
Final Project
Deadline: December 17, 2023
coded by: Justine Serdoncillo start 12/6/23
adapted by Ricardo
"""

#### IMPORT LIBRARIES ####
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

#### Residual Block
def res_block(in_channels, out_channels, strides, first=False):
    layers = []
    if first==False:
        layers.append(nn.BatchNorm2d(in_channels))
        layers.append(nn.ReLU())
    layers.append(nn.Conv2d(in_channels, out_channels,kernel_size=3, stride=strides[0],  padding=1))
    layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU())
    layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=strides[1],  padding=1))
    
    return nn.Sequential(*layers) 
        

class deepResUnet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.encoding1 = res_block(in_channels, 64, [1,1], first=True)
        self.encoding2 = res_block(64, 128, [2,1])
        self.encoding3 = res_block(128, 256, [2,1])
        
        self.bridge = res_block(256, 512, [2,1])
        
        self.decoding1 = res_block(512, 256, [1,1])
        self.decoding2 = res_block(256, 128, [1,1])
        self.decoding3 = res_block(128, 64, [1,1])
        self.upsamp = nn.Upsample(scale_factor=2, mode='bilinear')
        self.convlast = nn.Conv2d(64, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=1, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=1, stride=2)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=1, stride=2)
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=1)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=1)
        self.conv7 = nn.Conv2d(64, 128, kernel_size=1)
           
    def forward(self, x):
        # encoding
        y1 = self.encoding1(x) + self.conv1(x)
        y2 = self.encoding2(y1) + self.conv2(y1)
        y3 = self.encoding3(y2) + self.conv3(y2)
    
        # bridge
        y_bridge = self.bridge(y3) + self.conv4(y3)
        #print(y_bridge.shape)
        
        # decoding
        a = nn.Conv2d(512, 256, kernel_size=1)(self.upsamp(y_bridge))
        Y1 = torch.cat((a, y3), 1)
        Y1 = self.decoding1(Y1) + nn.Conv2d(512, 256, kernel_size=1)(Y1)
        #print(Y1.shape)
        
        a = nn.Conv2d(256, 128, kernel_size=1)(self.upsamp(Y1))
        #print(y2.shape)
        Y2 = torch.cat((a, y2), 1)
        Y2 = self.decoding2(Y2) + nn.Conv2d(256, 128, kernel_size=1)(Y2)
        #print(Y2.shape)
        
        
        a = nn.Conv2d(128, 64, kernel_size=1)(self.upsamp(Y2))
        #print(y1.shape)
        Y3 = torch.cat((a, y1), 1)
        Y3 = self.decoding3(Y3) + nn.Conv2d(128, 64, kernel_size=1)(Y3)
        #print(Y3.shape)
        
        y = self.convlast(Y3)
        y = self.sigmoid(y)
        
        return y

        

