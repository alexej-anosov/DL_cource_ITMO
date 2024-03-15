import torch
import torch.nn as nn
from collections import OrderedDict


class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, groups):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                        kernel_size=kernel_size, padding=padding,
                        stride=stride, groups=groups)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, input_tensor):
        x = self.conv(input_tensor)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class DepthwiseSeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super().__init__()
        self.dw_conv = BasicConvBlock(in_channels, in_channels,
                                 kernel_size=kernel_size, padding=padding,
                                 stride=stride, groups=in_channels)
        self.pw_conv = BasicConvBlock(in_channels, out_channels,
                                 kernel_size=1, padding=0,
                                 stride=1, groups=1)

    def forward(self, input_tensor):
        x = self.dw_conv(input_tensor)
        x = self.pw_conv(x)
        return x


class MobileNetV1(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

        self.conv1 = BasicConvBlock(3, 32, kernel_size=3, stride=2, padding=1, groups=1) 
        self.conv2 = DepthwiseSeparableConvBlock(32, 64, kernel_size=3, padding=1, stride=1)
        self.conv3 = DepthwiseSeparableConvBlock(64, 128, kernel_size=3, padding=1, stride=2)
        self.conv4 = DepthwiseSeparableConvBlock(128, 128, kernel_size=3, padding=1, stride=1)
        self.conv5 = DepthwiseSeparableConvBlock(128, 256, kernel_size=3, padding=1, stride=2)
        self.conv6 = DepthwiseSeparableConvBlock(256, 256, kernel_size=3, padding=1, stride=1)
        self.conv7 = DepthwiseSeparableConvBlock(256, 512, kernel_size=3, padding=1, stride=2)
        
        self.conv8 = DepthwiseSeparableConvBlock(512, 512, kernel_size=3, padding=1, stride=1)
        self.conv9 = DepthwiseSeparableConvBlock(512, 512, kernel_size=3, padding=1, stride=1)
        self.conv10 = DepthwiseSeparableConvBlock(512, 512, kernel_size=3, padding=1, stride=1)
        self.conv11 = DepthwiseSeparableConvBlock(512, 512, kernel_size=3, padding=1, stride=1)
        self.conv12 = DepthwiseSeparableConvBlock(512, 512, kernel_size=3, padding=1, stride=1)
        
        self.conv13 = DepthwiseSeparableConvBlock(512, 1024, kernel_size=3, padding=1, stride=2)
        self.conv14 = DepthwiseSeparableConvBlock(1024, 1024, kernel_size=3, padding=1, stride=1)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        self.fc = nn.Linear(1024, n_classes)
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x.view(-1, self.n_classes)

