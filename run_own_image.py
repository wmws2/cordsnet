import torch
import numpy as np
from torch.utils.data import DataLoader
from cordsnet import *
from utils import *
from PIL import Image

# set device to GPU
device = 'cuda'

# alpha is defined as the discretization time step divided by the neuron time constant 
# i.e. reciprocal of how many time steps are simulated for the duration of one neuron time constant
alpha = 0.2

# load dataset using torch, make sure the dataset is in the correct format
# ** THIS ONLY WORKS ON SQUARE IMAGES ** if your image is rectangular, pad with zeros first
# this returns a [1, 3, 224, 224] image
# feel free to stack a bunch of your own images
own_image = process_image(Image.open('example_image.png').convert('RGB')) 

# load model and pretrained weights (the publicly available model available on GitHub is trained on ImageNet)
model = cordsnet(dataset='imagenet',depth=8).to(device)
model.load_state_dict(torch.load('./cordsnetr8.pth'))
model.eval()

with torch.no_grad():

    own_image = own_image.to(device)

    # there are 8 layers in this model, labeled 0 to 7
    # the last 2 layers (6 and 7) are the ones analyzed in the paper
    # feel free to change the "layers" argument to get the rest of the activities
    activity = model.record(own_image, device, alpha, layers=[6,7]) 

# just to show how activity looks like, this gets the activity in layer 6
# the shape would be [200, number of images, C, H, W]
# for the sake of functioning like a CNN, the neurons are arranged in a "cuboid-like manner"
# so there are C x H x W neurons in total per layer
# the first 100 time steps are for the model to warm up
# the next 100 time steps are when the image is presented to the model
activity_in_layer_6 = activity[6]

        