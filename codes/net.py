"""
EE 5561 - Image Processing
Final Project
Deadline: December 17, 2023
coded by: Justine Serdoncillo start 12/6/23
adapted by Ricardo
"""

#### IMPORT LIBRARIES ####
import torch
import torch.nn as nn

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
        self.encoding1 = res_block(in_channels,  64, [1,1], first=True)
        self.encoding2 = res_block(         64, 128, [2,1])
        self.encoding3 = res_block(        128, 256, [2,1])
        
        self.bridge    = res_block(        256, 512, [2,1])
        
        self.decoding1 = res_block(        512, 256, [1,1])
        self.decoding2 = res_block(        256, 128, [1,1])
        self.decoding3 = res_block(        128,  64, [1,1])
        
        self.upsamp    = nn.Upsample(scale_factor=2, mode='bilinear')
        
        self.convlast  = nn.Conv2d(64, num_classes, kernel_size=1)
        self.sigmoid   = nn.Sigmoid()

        self.conv1 = nn.Conv2d(  3,  64, kernel_size=1)
        self.conv2 = nn.Conv2d( 64, 128, kernel_size=1, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=1, stride=2)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=1, stride=2)
        
        self.conv50 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv60 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv70 = nn.Conv2d(128,  64, kernel_size=1)
        self.conv51 = nn.Conv2d(256, 512, kernel_size=1)
        self.conv61 = nn.Conv2d(128, 256, kernel_size=1)
        self.conv71 = nn.Conv2d( 64, 128, kernel_size=1)
           
    def forward(self, x):
        
        # encoding
        y1 = self.encoding1(x)  + self.conv1(x)     
        y2 = self.encoding2(y1) + self.conv2(y1)
        y3 = self.encoding3(y2) + self.conv3(y2)
    
        # bridge
        y_bridge = self.bridge(y3) + self.conv4(y3)
        
        # decoding
        Y1 = torch.cat([self.conv70(self.upsamp(y_bridge)), y3], 1)
        Y1 = self.decoding1(Y1) + self.conv51(Y1)
        
        Y2 = torch.cat([self.conv70(self.upsamp(Y1)), y2], 1)
        Y2 = self.decoding2(Y2) + self.conv61(Y2)
        
        Y3 = torch.cat([self.conv70(self.upsamp(Y2)), y1], 1)
        Y3 = self.decoding3(Y3) + self.conv71(Y3)
        
        y = self.convlast(Y3)
        y = self.sigmoid(y)
        
        return y
    
#in_channels = 3  
#out_channels = 1 
#model = deepResUnet(in_channels, out_channels)

#print(model)

        

