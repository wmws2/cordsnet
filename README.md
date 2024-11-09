<h2 align="center">Recurrent neural network dynamical systems for biological vision</h2> 

### 1. Overview
Code and model checkpoint for the paper [[Recurrent neural network dynamical systems for biological vision]](https://openreview.net/forum?id=ZZ94aLbMOK) as presented in NeurIPS 2024. All code is written on PyTorch 2.2.0. 
- If you just want to immediately download and run a trained model, jump to [[3. Model Checkpoint]](#3-model-checkpoint).
- If you want details about training, jump to [[4. Model training]](#4-model-training).
- If you want the analysis toolkit, jump to [[5. Analysis toolkit]](#5-analysis-toolkit).
- Otherwise, read on below for an introduction to this work.

### 2. Introduction
<p align="center">
  <img src="/figure.png" width="600">
</p>

Convolutional neural networks (CNNs) are commonly seen in vision neuroscience literature due to their image processing capabilities. A typical convolutional layer in a CNN trained for image classification consists of a 2D-convolution, normalization, followed by a non-linear function:
```math
\mathbf{x}_l = (\sigma \circ \text{Norm} \circ \text{Conv})(\mathbf{x}_{l-1})
```

Recurrent neural networks (RNNs), specifically continuous-time dynamical RNNs, are widespread in the broader neuroscience literature due to the importance of recurrent dynamics in brain computations. The typical model obeys a differential equation:
```math
\mathbf{T}\frac{d\mathbf{r}}{dt} = - \mathbf{r} + \sigma\left( \mathbf{W}_\text{rec}\mathbf{r} + \mathbf{b} + \mathbf{W}_\text{inp}\mathbf{h}_\text{inp} \right)
```

This leads to a very interesting situation where RNNs and CNNs are both proposed models of the visual system. As a result, CNNs with recurrence have been proposed in the past, but here we take it a step further and introduce the [Co]nvolutional [R]NN [D]ynamical [S]ystem, or CordsNet, which introduces continuous-time recurrent dynamics into CNNs:
```math
\mathbf{T}\frac{d\mathbf{r}_l}{dt} = - \mathbf{r}_l + \sigma\left( \text{Conv}_\text{rec}(\mathbf{r}_l) + \mathbf{b} + \text{Conv}_\text{inp}(\mathbf{h}_\text{inp}) \right)
```

Please check out our paper for more details! 

### 3. Model checkpoint
In this repository, we have uploaded a training checkpoint of CordsNet-R8 pretrained on ImageNet along with minimalistic code required to run the model. All required files are in the main directory.

#### 3.1 Files needed
- <code>main.py</code> is the script to run and experiment which imports or loads the other files below. <br>
- <code>cordsnet.py</code> contains the model class. <br>
- <code>utils.py</code> contains all image preprocessing functions as well as other utilities. <br>
- <code>cordsnetr8.pth</code> contains the model state dictionary of the trained model. <br>
- If you wish to work with ImageNet, you will need to download and unpack <code>ILSVRC2012_img_train.tar</code> and <code>ILSVRC2012_img_val.tar</code>. We refrain from distributing the dataset here, but a quick Google search should get you the link or torrent to those files. Other datasets in this work, namely MNIST, F-MNIST and CIFAR-10/100 will automatically download if you do not have them when you run our code.

#### 3.2 Description
In the current state, <code>main.py</code> is written such that images from the ImageNet dataset are passed to the model as input, but you may simply replace <code>inputs</code> with your own desired images. The <code>forward</code> method in the <code>cordsnet</code> class receives input with format <code>[inputs,device,alpha]</code>, where <code>inputs</code> is the array of images, <code>device</code> is the device to run the model on (ideally a GPU), and <code>alpha</code> is the time discretization divided by the time constant. It was set to 0.2 during training, representing 2 ms time steps and 10 ms time constants. <code>cordsnet.py</code> is currently written to give the pre-softmax layer as output, but feel free to make edits if you want to extract neural activities <code>rs</code> at the current time step.  

### 4. Model training
The working directory here is <code>./training/</code>. If you just want to see how the model is simulated and how the loss is calculated, refer to <code>class_rnn_5_final.py</code>. Otherwise, we provide all the code required to train our models here. Refer to the files in the main directory for annotations that will help with understanding the code here. Note that most models were trained on multiple nodes, each with multiple GPUs; the code is specifically written to do that on our clusters and will not user-friendly. We recommend writing your own code for distributed training tailored to your own available hardware. 

#### 4.1 Files needed
- <code>class_rnn_1_compatibility.py</code>/<code>main_training_1_compatibility.py</code> are the files required to train the equivalent CNN in step 1 of our proposed method. <br>
- <code>class_rnn_1_folding.py</code>/<code>main_training_1_folding</code> are used to perform batch normalization folding. <br>
- <code>class_rnn_2_inverse.py</code>/<code>main_training_2_inverse.py</code> are used to initialize the linear model in step 2 of our proposed method. <br>
- <code>class_rnn_3_linear.py</code>/<code>main_training_3_linear.py</code> are the files required to train the linear model. <br>
- <code>class_rnn_4_parametric.py</code>/<code>main_training_4_parametric.py</code> are the files required to anneal the linear model into a non-linear one in step 3 of our proposed method. <br>
- <code>class_rnn_5_final.py</code>/<code>main_training_5_finetune.py</code> are the files required to finetune the model after initializing weights using our proposed method. <br>
- If you wish to work with ImageNet, you will need to download and unpack <code>ILSVRC2012_img_train.tar</code> and <code>ILSVRC2012_img_val.tar</code>. We refrain from distributing the dataset here, but a quick Google search should get you the link or torrent to those files. Other datasets in this work, namely MNIST, F-MNIST and CIFAR-10/100 will automatically download if you do not have them when you run our code.

#### 4.2 Description
We provide two ways to train the model: one using our proposed method to initialize the weights and then fine tune, and another by training directly from random initialization. To use our proposed method, run the scripts in numerical order. To train a model directly after randomly initializing model weights, run <code>main_training_5_finetune.py</code> without loading pretrained weights. 

### 5. Analysis toolkit
The working directory here is <code>./analysis/</code>. We provide the code for numerically stable implementations of Arnoldi iteration and power iteration specifically tailored for convolutional structures. 

#### 5.1 Files needed
- <code>compute_eigenvalues.py</code> is the script to compute eigenvalues of convolutions using Arnoldi iteration. <br>
- <code>compute_eigenvector.py</code> is the script to compute the top eigenvector of convolutions using power iteration. <br>

#### 5.2 Description
In both files, we first convert a convolutional operation into a linear 2-D weight matrix, and perform eigendecomposition directly. You may use them to analyze smaller CNNs. In the second half of both files, we provide the code to the iterative algorithms described above. In <code>compute_eigenvalues.py</code>, we perform a slow but stable 100-step QR decomposition after Arnoldi iterations to ensure stability. Based on our testing, this still runs within a tolerable time window of <10s on our largest models. In <code>compute_eigenvector.py</code>, we perform parallelized power iteration leveraging the memory of our GPUs, but you may set <code>trials = 1</code> and still get consistent results.

### 6. Citation
```
@inproceedings{soo2024recurrent,
 author = {Soo, Wayne W.M. and Battista, Aldo and Radmard, Puria and Wang, Xiao-Jing},
 booktitle = {Advances in Neural Information Processing Systems},
 publisher = {Curran Associates, Inc.},
 title = {Recurrent neural network dynamical systems for biological vision},
 volume = {37},
 year = {2024}
}
```
