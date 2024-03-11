import torch
import torch.nn as nn
from collections import OrderedDict


class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, groups, bias):
        super(BasicConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                        kernel_size=kernel_size, padding=padding,
                        stride=stride, groups=groups, bias=bias)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6()
    
    def forward(self, input_tensor):
        x = self.conv(input_tensor)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class BlockForStrideOne(nn.Module):
    def __init__(self, in_channels, out_channels, expansion):
        super().__init__()
        self.expansion_conv = BasicConvBlock(in_channels, expansion*in_channels, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        self.dw_conv = BasicConvBlock(expansion*in_channels, expansion*in_channels,
                                 kernel_size=3, padding=1,
                                 stride=1, groups=in_channels, bias=False)
        self.pw_conv = BasicConvBlock(expansion*in_channels, out_channels,
                                 kernel_size=1, padding=0,
                                 stride=1, groups=1, bias=False)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, input_tensor):
        x = self.expansion_conv(input_tensor)
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        
        if self.in_channels == self.out_channels:
            x += input_tensor
            
        return x


class BlockForStrideTwo(nn.Module):
    def __init__(self, in_channels, out_channels, expansion):
        super().__init__()
        self.expansion_conv = BasicConvBlock(in_channels, expansion*in_channels, kernel_size=1, padding=0, stride=1, groups=1, bias=False)
        self.dw_conv = BasicConvBlock(expansion*in_channels, expansion*in_channels,
                                 kernel_size=3, padding=1,
                                 stride=2, groups=in_channels, bias=False)
        self.pw_conv = BasicConvBlock(expansion*in_channels, out_channels,
                                 kernel_size=1, padding=0,
                                 stride=1, groups=1, bias=False)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, input_tensor):
        x = self.expansion_conv(input_tensor)
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        
        if self.in_channels == self.out_channels:
            x += input_tensor
            
        return x



class MobileNetV2(nn.Module):
    def __init__(self, model_configuration, in_channels=3, n_classes=1000, width_multiplier=1):
        super().__init__()
        self.n_classes =  n_classes
        self.model_configuration = model_configuration
        self.width_multiplier = width_multiplier
        
        self.input_conv = BasicConvBlock(in_channels, int(width_multiplier*32), kernel_size=3, padding=1, stride=2, groups=1, bias=False)
  
        self.in_channels = int(width_multiplier*32)
        
        self.configurated_layers = self.ModelConfigurator()
        
        self.feature = BasicConvBlock(self.in_channels, int(width_multiplier*1280), kernel_size=1, padding=0, stride=1, groups=1, bias=False)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))

        self.dropout = nn.Dropout2d()
        self.output_conv = nn.Conv2d(int(width_multiplier*1280), self.n_classes, kernel_size=1)

    def forward(self, input_tensor):
        x = self.input_conv(input_tensor)
        x = self.configurated_layers(x)
        x = self.feature(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = self.output_conv(x)
        return x.view(-1,  self.n_classes)


    def ModelConfigurator(self):
        itemIndex = 0
        block = OrderedDict()
        for layer_configuration in self.model_configuration:
            t, out_channels, n, stride = layer_configuration
            out_channels = int(self.width_multiplier*out_channels)
            if stride == 1:
                for item_num in range(n):
                    block[str(itemIndex)+"_"+str(item_num)] = BlockForStrideOne(self.in_channels, out_channels, t)
                    self.in_channels = out_channels
            elif stride == 2:
                block[str(itemIndex)+"_"+str(0)] = BlockForStrideTwo(self.in_channels, out_channels, t)
                self.in_channels = out_channels
                for item_num in range(1,n):
                    block[str(itemIndex)+"_"+str(item_num)] = BlockForStrideOne(self.in_channels, out_channels, t)
            itemIndex += 1
        return nn.Sequential(block)