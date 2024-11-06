import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
import time

class step2cordsnetinverse(nn.Module):

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
        conv_bias = []
        area_area = []
        for i in range(self.depth):
            area_conv.append(nn.utils.parametrizations.weight_norm(nn.Conv2d(channels[i+1],channels[i+1],kernel_size=3,stride=1,padding=1,bias=False)))
            conv_bias.append(nn.Parameter(torch.zeros(channels[i+1],sizes[i],sizes[i]), requires_grad=True))
            area_area.append(nn.utils.parametrizations.weight_norm(nn.Conv2d(channels[i],channels[i+1],kernel_size=3,stride=strides[i],padding=1,bias=False)))
        self.area_conv = nn.ParameterList(area_conv)
        self.conv_bias = nn.ParameterList(conv_bias)
        self.area_area = nn.ParameterList(area_area)
        
        skip_area = []
        for i in range(self.blockdepth):
            skip_area.append(nn.utils.parametrizations.weight_norm(nn.Conv2d(skipchannels[i],skipchannels[i+1],kernel_size=1,stride=2,padding=0,bias=False)))
        self.skip_area = nn.ParameterList(skip_area)

        # target model

        self.target_inp_conv = nn.utils.parametrizations.weight_norm(nn.Conv2d(N_channel,64,kernel_size=7,stride=2,padding=3, bias=False))
        self.target_inp_avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.target_inp_skip = nn.utils.parametrizations.weight_norm(nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1, bias=False))   

        target_area_conv = []
        target_area_bias = []
        target_conv_bias = []
        target_area_area = []

        for i in range(self.depth):
            target_area_conv.append(nn.utils.parametrizations.weight_norm(nn.Conv2d(channels[i+1],channels[i+1],kernel_size=3,stride=1,padding=1,bias=False)))
            target_area_bias.append(nn.Parameter(torch.zeros(channels[i+1],sizes[i],sizes[i]), requires_grad=True))
            target_conv_bias.append(nn.Parameter(torch.zeros(channels[i+1],sizes[i],sizes[i]), requires_grad=True))
            target_area_area.append(nn.utils.parametrizations.weight_norm(nn.Conv2d(channels[i],channels[i+1],kernel_size=3,stride=strides[i],padding=1,bias=False)))
        self.target_area_conv = nn.ParameterList(target_area_conv)
        self.target_area_bias = nn.ParameterList(target_area_bias)
        self.target_conv_bias = nn.ParameterList(target_conv_bias)
        self.target_area_area = nn.ParameterList(target_area_area)
        
        target_skip_area = []
        for i in range(self.blockdepth):
            target_skip_area.append(nn.utils.parametrizations.weight_norm(nn.Conv2d(skipchannels[i],skipchannels[i+1],kernel_size=1,stride=2,padding=0,bias=False)))
        self.target_skip_area = nn.ParameterList(target_skip_area)

        self.out_avgpool = nn.AdaptiveAvgPool2d(1)
        self.out_flatten = nn.Flatten()
        self.out_fc = nn.utils.parametrizations.weight_norm(nn.Linear(channels[self.depth],N_class,bias=True))

    def forward(self,inputs,rank,areas):

        current_batch_size = inputs.size()[0]
        channels = [64,64,128,128,256,256,512,512]
        sizes = [56,56,28,28,14,14,7,7]

        x0 = []
        for j in range(self.depth):
            x0.append(torch.zeros(current_batch_size,channels[j],sizes[j],sizes[j]).to(rank)) 
        errors = torch.zeros(self.depth).to(rank)
        eigens = torch.zeros(self.depth).to(rank)

        fixed = self.cnnactivity(x0,inputs)
        inp = self.target_inp_avgpool(self.target_inp_conv(inputs))
        prernninputs = []
        prernninputs.append(inp)
        prernninputs.append(fixed[0] + self.inp_skip(inp))
        if self.depth > 2:
            prernninputs.append(fixed[1] + fixed[0])
            prernninputs.append(fixed[2] + self.skip_area[0](fixed[1]))
        if self.depth > 4:
            prernninputs.append(fixed[3] + fixed[2])
            prernninputs.append(fixed[4] + self.skip_area[1](fixed[3]))
        if self.depth > 6:
            prernninputs.append(fixed[5] + fixed[4])
            prernninputs.append(fixed[6] + self.skip_area[2](fixed[5]))
        rnninputs = [self.target_area_area[i](prernninputs[i]) for i in range(self.depth)]

        for i in areas:
            targetr = self.target_area_conv[i](rnninputs[i]) + self.target_area_conv[i](self.target_area_bias[i]) + self.target_conv_bias[i]

            steadyr = self.conv_bias[i] + rnninputs[i]
            steadyr2 = self.conv_bias[i] + rnninputs[i]
            for t in range(3):
                steadyr = self.area_conv[i](steadyr) + self.conv_bias[i] + rnninputs[i]
                steadyr2 = self.area_conv[i](steadyr2)
            errors[i] += ((targetr - steadyr)**2).mean()
            eigens[i] += (steadyr2**2).mean()
        
        return errors, eigens

    def cnnactivity(self,x0,img):

        x = [xx.clone() for xx in x0]

        inp  = self.target_inp_avgpool(self.target_inp_conv(img))

        x[0] = self.target_area_area[0](inp) + self.target_area_bias[0]
        x[0] = self.relu(self.target_area_conv[0](x[0]) + self.target_conv_bias[0])
        x[1] = self.target_area_area[1](x[0] + self.target_inp_skip(inp)) + self.target_area_bias[1]
        x[1] = self.relu(self.target_area_conv[1](x[1]) + self.target_conv_bias[1])

        for i in range(self.blockdepth):
            x[2+2*i] = self.target_area_area[2+2*i](x[1+2*i] + x[2*i]) + self.target_area_bias[2+2*i]
            x[2+2*i] = self.relu(self.target_area_conv[2+2*i](x[2+2*i]) + self.target_conv_bias[2+2*i])
            x[3+2*i] = self.target_area_area[3+2*i](x[2+2*i]+ self.target_skip_area[i](x[1+2*i])) + self.target_area_bias[3+2*i]
            x[3+2*i] = self.relu(self.target_area_conv[3+2*i](x[3+2*i]) + self.target_conv_bias[3+2*i])

        return x