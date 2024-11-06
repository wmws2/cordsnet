import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
import time

class cordsnet(nn.Module):

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

        self.out_avgpool = nn.AdaptiveAvgPool2d(1)
        self.out_flatten = nn.Flatten()
        self.out_fc = nn.utils.parametrizations.weight_norm(nn.Linear(channels[self.depth],N_class,bias=True))

        self.criterion = nn.CrossEntropyLoss(reduction='sum')

    def forward(self,inputs,labels,rank,alpha):

        current_batch_size = inputs.size()[0]
        channels = [64,64,128,128,256,256,512,512]
        sizes = [56,56,28,28,14,14,7,7]

        x0 = []
        rs = []
        for j in range(self.depth):
            x0.append(torch.zeros(current_batch_size,channels[j],sizes[j],sizes[j]).to(rank)) 
            rs.append(torch.zeros(current_batch_size,channels[j],sizes[j],sizes[j]).to(rank))

        timesteps = 100
        loss = torch.zeros([30]).to(rank)
        accs = torch.zeros([30]).to(rank)

        with torch.no_grad():
            for t in range(timesteps):
                for j in range(self.depth-1, -1, -1):
                    rs[j] = self.rnn(j,rs,inputs*0,alpha)
            baseline = [r.clone() for r in rs]

        for t in range(timesteps):
            for j in range(self.depth-1, -1, -1):
                rs[j] = self.rnn(j,rs,inputs,alpha)
            if t>=70:
                out = self.out_avgpool(self.relu(rs[self.depth-1])+self.relu(rs[self.depth-2]))
                out = self.out_flatten(out)
                out = self.out_fc(out)
                loss[t-70] += self.criterion(out,labels)/current_batch_size
                accs[t-70] += (out.argmax(1) == labels.argmax(1)).sum().item()/(current_batch_size)
        
        spontaneous = 0.
        for t in range(timesteps):
            for j in range(self.depth-1, -1, -1):
                rs[j] = self.rnn(j,rs,inputs*0,alpha)
                if t>=90:
                    spontaneous += torch.mean((rs[j]-baseline[j])**2)/(self.depth*10)

        return loss, accs, spontaneous

    def rnn(self,area,r,img,alpha):

        inp = self.inp_avgpool(self.inp_conv(img))
        if area == 0:
            areainput = inp
        if area == 1:
            areainput = self.relu(r[0]) + self.inp_skip(inp)
        if area == 2:
            areainput = self.relu(r[1]) + self.relu(r[0])
        if area == 3:
            areainput = self.relu(r[2]) + self.skip_area[0](self.relu(r[1]))
        if area == 4:
            areainput = self.relu(r[3]) + self.relu(r[2])
        if area == 5:
            areainput = self.relu(r[4]) + self.skip_area[1](self.relu(r[3]))
        if area == 6:
            areainput = self.relu(r[5]) + self.relu(r[4])
        if area == 7:
            areainput = self.relu(r[6]) + self.skip_area[2](self.relu(r[5]))
        
        r[area]  = (1-alpha)*r[area] + alpha*self.relu(self.area_conv[area](r[area]) + self.conv_bias[area] + self.area_area[area](areainput))
        return r[area]