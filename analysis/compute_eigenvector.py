import torch
import torch.nn as nn
import numpy as np

def norm(popvector):
    return torch.sqrt(torch.sum(torch.square(popvector),axis=(1,2,3),keepdim=True))

with torch.no_grad():

    # as an example, we randomly initialize a convolution operation
    # the input and output layers have 10 channels and size 10 x 10
    channels = 10
    size = 10
    convlayer = nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1,bias=False)

    # converting the convolution operation to a 2-D weight matrix
    # if we flatten the input layer to get a vector of length population = channels x size x size
    # then the weight matrix that operates on this vector would have size population x population
    population = channels*size*size
    weightmatrix = torch.zeros([population,population])
    for i in range(population):
        testvector = torch.zeros([channels*size*size])
        testvector[i] = 1
        testvector = testvector.view(1,channels,size,size)
        weightmatrix[:,i] = convlayer(testvector).flatten()

    # compute the eigenvalues directly as the ground truth for comparison with our algorithm later
    testvals, testvecs = torch.linalg.eig(weightmatrix)
    testvec = testvecs[:,0].detach().numpy()
    testvec = testvec/np.linalg.norm(testvec)

    # performing the power iteration across parallelized trials
    iterations = 10000
    trials = 10
    popvector = torch.rand(trials,channels,size,size)
    popvector = popvector/norm(popvector)

    for k in range(iterations):
        popvector = convlayer(popvector)
        popvector = popvector/norm(popvector)

    popvector = torch.mean(popvector,axis=0)
    popvector = popvector.detach().numpy().flatten()
    vec = popvector/np.linalg.norm(popvector)

    # you can then compare testvec and vec to make sure that they match
    # vec can either be equal to testvec or -1*testvec
    # the power iteration only converges if the spectral gap is sufficiently large, which is not that common
    # run the script multiple times to get an idea of how often this works


