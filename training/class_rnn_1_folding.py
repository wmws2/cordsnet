import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class step1cordsnetfolded(nn.Module):

    def __init__(self, dataset, depth):

        super().__init__()

        if dataset in ['imagenet']:
            N_class = 1000
            N_channel = 3

        elif dataset in ['cifar100']:
            N_class = 100
            N_channel = 3

        elif dataset in ['cifar10']:
            N_class = 10
            N_channel = 3

        elif dataset in ['mnist','fashionmnist']:
            N_class = 10
            N_channel = 1

        self.depth = depth
        self.blockdepth = int(depth/2-1)
        self.relu = nn.ReLU(inplace=False)

        self.inp_conv = nn.utils.parametrizations.weight_norm(nn.Conv2d(N_channel,64,kernel_size=7,stride=2,padding=3, bias=False))
        self.inp_avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.inp_skip = nn.utils.parametrizations.weight_norm(nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1, bias=False))    

        channels = [64,64,64,128,128,256,256,512,512]
        sizes = [56,56,28,28,14,14,7,7]
        strides = [1,1,2,1,2,1,2,1]
        skipchannels = [64,128,256,512]

        area_conv = []
        area_bias = []
        conv_bias = []
        area_area = []
        for i in range(self.depth):
            area_conv.append(nn.utils.parametrizations.weight_norm(nn.Conv2d(channels[i+1],channels[i+1],kernel_size=3,stride=1,padding=1,bias=False)))
            area_bias.append(nn.Parameter(torch.zeros(channels[i+1],sizes[i],sizes[i]), requires_grad=True))
            conv_bias.append(nn.Parameter(torch.zeros(channels[i+1],sizes[i],sizes[i]), requires_grad=True))
            area_area.append(nn.utils.parametrizations.weight_norm(nn.Conv2d(channels[i],channels[i+1],kernel_size=3,stride=strides[i],padding=1,bias=False)))
        self.area_conv = nn.ParameterList(area_conv)
        self.area_bias = nn.ParameterList(area_bias)
        self.conv_bias = nn.ParameterList(conv_bias)
        self.area_area = nn.ParameterList(area_area)
        
        skip_area = []
        for i in range(self.blockdepth):
            skip_area.append(nn.utils.parametrizations.weight_norm(nn.Conv2d(skipchannels[i],skipchannels[i+1],kernel_size=1,stride=2,padding=0,bias=False)))
        self.skip_area = nn.ParameterList(skip_area)

        self.out_avgpool = nn.AdaptiveAvgPool2d(1)
        self.out_flatten = nn.Flatten()
        self.out_fc = nn.utils.parametrizations.weight_norm(nn.Linear(channels[self.depth],N_class,bias=True))

    def forward(self,x,img):

        inp  = self.inp_avgpool(self.inp_conv(img))

        # layer 1
        x[0] = self.area_area[0](inp) + self.area_bias[0]
        x[0] = self.relu(self.area_conv[0](x[0]) + self.conv_bias[0])
        x[1] = self.area_area[1](x[0] + self.inp_skip(inp)) + self.area_bias[1]
        x[1] = self.relu(self.area_conv[1](x[1]) + self.conv_bias[1])

        # layers 2,3,4
        for i in range(self.blockdepth):
            x[2+2*i] = self.area_area[2+2*i](x[1+2*i] + x[2*i]) + self.area_bias[2+2*i]
            x[2+2*i] = self.relu(self.area_conv[2+2*i](x[2+2*i]) + self.conv_bias[2+2*i])
            x[3+2*i] = self.area_area[3+2*i](x[2+2*i]+ self.skip_area[i](x[1+2*i])) + self.area_bias[3+2*i]
            x[3+2*i] = self.relu(self.area_conv[3+2*i](x[3+2*i]) + self.conv_bias[3+2*i])

        out = self.out_avgpool(x[self.depth-1]+x[self.depth-2])
        out = self.out_flatten(out)
        out = self.out_fc(out)

        return out, x