import torch
import numpy as np
from torch.utils.data import DataLoader
from cordsnet import *
from utils import *

# set device to GPU
device = 'cuda'

# alpha is defined as the discretization time step divided by the neuron time constant 
# i.e. reciprocal of how many time steps are simulated for the duration of one neuron time constant
alpha = 0.2

# load dataset using torch, make sure the dataset is in the correct format
dataset = 'imagenet'
getdataset = load_dataset(dataset,'test')
dataloader = DataLoader(getdataset, batch_size=64)

# load model and pretrained weights
model = cordsnet(dataset=dataset,depth=8).to(device)
model.load_state_dict(torch.load('./cordsnetr8.pth'))
model.eval()

with torch.no_grad():
    for i, data in enumerate(dataloader):

        # load a single batch from dataloader
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # get the output of the model pre-softmax
        out = model(inputs,labels,device,alpha)
        