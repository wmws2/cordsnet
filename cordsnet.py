import torch
import torch.nn as nn

class cordsnet(nn.Module):

    def __init__(self, dataset, depth):

        super().__init__()

        # the dataset determines how many units there are at the final classification layer 
        # and whether the input has 1 channel (black/white) or 3 channels (RGB)

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
        
        # defining depth as number of recurrent layers, and blockdepth as 
        self.depth = depth
        self.blockdepth = int(depth/2-1)
        self.relu = nn.ReLU(inplace=False)
        channels = [64,64,64,128,128,256,256,512,512]
        sizes = [56,56,28,28,14,14,7,7]
        strides = [1,1,2,1,2,1,2,1]
        skipchannels = [64,128,256,512]

        # define the trainable parameters of the model
        # most parameters are parameterized with weight normalization to control the magnitude and direction independently

        # input layer
        # inp_conv represents the a convolution on the input image to the first layer
        # inp_skip represents the a convolution on the input image to the second layer
        self.inp_conv = nn.utils.parametrizations.weight_norm(nn.Conv2d(N_channel,64,kernel_size=7,stride=2,padding=3, bias=False))
        self.inp_avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.inp_skip = nn.utils.parametrizations.weight_norm(nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1, bias=False))   

        # middle layers
        # area_conv represents the recurrent convolution within an area
        # conv_bias represents the bias term 
        # area_area represents the convolution from one area to the next area
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
        
        # residual connections
        skip_area = []
        for i in range(self.blockdepth):
            skip_area.append(nn.utils.parametrizations.weight_norm(nn.Conv2d(skipchannels[i],skipchannels[i+1],kernel_size=1,stride=2,padding=0,bias=False)))
        self.skip_area = nn.ParameterList(skip_area)

        # final layers
        self.out_avgpool = nn.AdaptiveAvgPool2d(1)
        self.out_flatten = nn.Flatten()
        self.out_fc = nn.utils.parametrizations.weight_norm(nn.Linear(channels[self.depth],N_class,bias=True))

    def forward(self,inputs,labels,rank,alpha):

        current_batch_size = inputs.size()[0]
        channels = [64,64,128,128,256,256,512,512]
        sizes = [56,56,28,28,14,14,7,7]

        # begin by defining the activations within the CNN
        rs = []
        for j in range(self.depth):
            rs.append(torch.zeros(current_batch_size,channels[j],sizes[j],sizes[j]).to(rank))

        timesteps = 100

        # running the network first without any input so that it reaches a steady-state spontaneous activity level
        with torch.no_grad():
            for t in range(timesteps):
                for j in range(self.depth-1, -1, -1):
                    rs[j] = self.rnn(j,rs,inputs*0,alpha)

        # running the network with the desired input
        for t in range(timesteps):
            for j in range(self.depth-1, -1, -1):
                rs[j] = self.rnn(j,rs,inputs,alpha)

        # computing the output
        out = self.out_avgpool(self.relu(rs[self.depth-1])+self.relu(rs[self.depth-2]))
        out = self.out_flatten(out)
        out = self.out_fc(out)

        return out

    def rnn(self,area,r,img,alpha):

        # simulating one area by one time step
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
        
        # RNN dynamics
        r[area]  = (1-alpha)*r[area] + alpha*self.relu(self.area_conv[area](r[area]) + self.conv_bias[area] + self.area_area[area](areainput))
        return r[area]