import torch
import torch.nn as nn
import numpy as np

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
    testvals = torch.linalg.eigvals(weightmatrix)
    testvals = testvals.detach().numpy()

    # performing the arnoldi iteration
    iterations = 300
    popvectors = torch.zeros(iterations+1,1,channels,size,size)
    popvector = torch.rand(1,channels,size,size)
    popvectors[0] = popvector/torch.norm(popvector)
    H = torch.zeros([iterations+1,iterations])

    for k in range(iterations):
        v = convlayer(popvectors[k])
        for l in range(k+1):
            H[l,k] = torch.sum(popvectors[l]*v)
            v = v - H[l,k] * popvectors[l]
        H[k+1,k] = torch.norm(v)
        popvectors[k+1] = v/H[k+1,k]
        A = H[:k+1,:k+1].clone()

    # computing the eigenvalues of A, which has the same eigenvalues as the convolution operation
    # this is done using QR algorithm with a high number of iterations for numerical stability
    for l in range(5000):
        Q,R = torch.linalg.qr(A)
        A = R@Q
    vals = torch.linalg.eigvals(A)

    # you can then compare testvals and vals to make sure that they match
    # if not, increase the number of iterations for both Arnoldi and QR