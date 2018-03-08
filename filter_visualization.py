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
#from baseline_cnn import twoLayeredCNN
#from /../pytorch-cnn-visualizations/src/cnn_layer_visualization import CNNLayerVisualization

def imshow(img):
    img = img *0.1307 + 0.3081     # unnormalize for MNIST
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def show(images):
    imshow(torchvision.utils.make_grid(images))    

class twoLayeredCNN(nn.Module):
    def __init__(self, filtersize):
        super(twoLayeredCNN, self).__init__()
        self.final_layer_dim = (7-np.int(filtersize/1.7))**2   # works for filtersizes 3,5,7
        # first layer 
        self.conv1 = nn.Conv2d(3, 48, filtersize) # TODO maybe change filter size
        self.batchnorm = nn.BatchNorm2d(48)   # is 2d the right one?
        self.pool = nn.MaxPool2d(2)
        self.prelu = nn.PReLU()
        # second layer
        self.conv2 = nn.Conv2d(48, 16, filtersize)
        self.batchnorm2 = nn.BatchNorm2d(16) 
        self.prelu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(2)
         
        self.fc1 = nn.Linear(16 * self.final_layer_dim, 10)

    def forward(self, x):
        x = self.conv1(x)
        #print('x after conv ', x.size())
        x = self.batchnorm(x)
        #print('batchnorm ', x.size())
        #x = F.prelu(self.nptn(x), 0.1) 
        x = self.pool(self.prelu(x))
        #print('shape first layer ', x.size())
        x = self.batchnorm2(self.conv2(x))
        #print('after batchnorm 2 ', x.size())
        x = self.pool2(self.prelu2(x))
        #print('shape second layer ', x.size())
        
        x = x.view(-1, 16 * self.final_layer_dim)
        #print('shape second layer ', x.size())
        x = F.log_softmax(self.fc1(x), dim=1)
        #print('after softmax ', x.size())
        return x
    
    

#net = twoLayeredCNN(5)
net = torch.load( 'CNN_test_model.final_CNN_model')