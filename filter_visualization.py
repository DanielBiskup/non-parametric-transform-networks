#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 11:07:17 2018

Load a trained model
"""


import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from network import twoLayeredNPTN
from network import threeLayeredNPTN
from network import twoLayeredCNN
from network import threeLayeredCNN
import random
#from baseline_cnn import twoLayeredCNN
#from /../pytorch-cnn-visualizations/src/cnn_layer_visualization import CNNLayerVisualization
from scipy.ndimage.interpolation import rotate


# for showing images
def imshow_mnist(img):
    img = img *0.1307 + 0.3081     # unnormalize for MNIST
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def show_mnist(img):
    imshow_mnist(torchvision.utils.make_grid(img))
    #npimg = img.numpy()
    #plt.imshow(np.transpose(npimg, (1, 2, 0)))
    

def imshow_cifar(img):
    img = img / 2 + 0.5     # unnormalize CIFAR
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def show_cifar(img):
    imshow_cifar(torchvision.utils.make_grid(img))    
    
    
# for showing filters
def plot_kernels(tensor, num_cols=6): # plots all kernels (can take a while)
    if not tensor.ndim==4:
        raise Exception("assumes a 4D tensor")
    #if not tensor.shape[-1]==3:
    #    raise Exception("last dim needs to be 3 to plot")
    #num_cols = tensor.shape[1]
    num_kernels = tensor.shape[0] * tensor.shape[1]
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    counter = 0
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):        
            ax1 = fig.add_subplot(num_rows,num_cols,counter+1)
            ax1.imshow(tensor[i][j])
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            counter += 1
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    

#plot a few kernels from a model
def example_plots_cnn(net, layer=2, num_samples=5):
    if layer == 1:
        t = net.conv1.weight.data.cpu().numpy()[:1, :num_samples]
    if layer == 2:    
        t = net.conv2.weight.data.cpu().numpy()[:1, :num_samples]
    if layer == 3:    
        t = net.conv3.weight.data.cpu().numpy()[:1, :num_samples]
    
    plot_kernels(t, num_cols=5)
  
    
def example_plots_nptn(net, layer=2, num_samples=5): # TODO implement random samples, random=False): 

    if layer == 1:
        g = net.nptn.G
        #num_filters = net.nptn.conv1.weight.shape[0]
        #if random:
        #   start = 
        t = net.nptn.conv1.weight.data.cpu().numpy()[:num_samples*g]       
    if layer == 2:    
        g = net.nptn2.G
        t = net.nptn2.conv1.weight.data.cpu().numpy()[:num_samples*g]    
    if layer == 3:    
        g = net.nptn3.G
        t = net.nptn3.conv1.weight.data.cpu().numpy()[:num_samples*g]       
    
    plot_kernels(t, num_cols=g)
  

#cnn_net = torch.load( '2018_03_09_11_35_25_mnist_1M_cnn_2layers36N1_16N2_5Kernel.final_model') # models run with CUDA also need CUDA for loading them again    
#example_plots_cnn(cnn_net)


