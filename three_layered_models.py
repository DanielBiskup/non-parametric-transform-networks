#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 14:50:28 2018

three layered networks
"""


import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from network import NPTN


class threeLayeredNPTN(nn.Module):
    def __init__(self, filtersize=5, G=3, n1=48, n2=48, n3=16, input_channel=3):
        super(threeLayeredNPTN, self).__init__()
        self.n3 = n3
        padding = int(filtersize/2) # needed if you want to use maxpooling 3 times
        if input_channel==3: # CIFAR
            self.final_layer_dim = 8*8  # for image size of 32x32
        else:
            self.final_layer_dim = 7*7  # TODO correct?
        # first layer 
        self.nptn = NPTN(input_channel, n1, G, filtersize, padding=padding)
        self.batchnorm = nn.BatchNorm2d(n1)   # is 2d the right one? ---> yes
        self.pool = nn.MaxPool2d(2)
        self.prelu = nn.PReLU()
        # second layer
        self.nptn2 = NPTN(n1, n2, G, filtersize, padding=padding)
        self.batchnorm2 = nn.BatchNorm2d(n2) 
        self.prelu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(2)
        #third layer 
        self.nptn3 = NPTN(n2, n3, G, filtersize, padding=padding)
        self.batchnorm3 = nn.BatchNorm2d(n3)
        self.prelu3 = nn.PReLU()
        #self.pool3 = nn.MaxPool2d(2)
        
        self.fc1 = nn.Linear(n3 * self.final_layer_dim, 10)

    def forward(self, x):
        # first layer
        x = self.batchnorm(self.nptn(x))
        #print('batchnorm ', x.size())
        x = self.pool(self.prelu(x))
        #print('shape first layer ', x.size())
        
        # second layer
        x = self.batchnorm2(self.nptn2(x))
        #print('after batchnorm 2 ', x.size())
        x = self.pool2(self.prelu2(x))
        #print('shape second layer ', x.size())
        
        # third layer
        x = self.batchnorm3(self.nptn3(x))
        #print('after batchnorm 3 ', x.size())
        #x = self.pool3(self.prelu3(x))
        #print('shape third layer ', x.size())
        
        x = x.view(-1, self.n3 * self.final_layer_dim)
        #print('shape third layer after view', x.size())
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)
        #print('after softmax ', x.size())
        return x
    

class threeLayeredCNN(nn.Module):
    def __init__(self, filtersize=5, n1=89, n2=89, n3=16, input_channel=3):
        super(threeLayeredCNN, self).__init__()
        self.n3 = n3
        padding = int(filtersize/2) # needed if you want to use maxpooling 3 times
        if input_channel==3: # CIFAR
            self.final_layer_dim = 8*8  # for image size of 32x32
        else:
            self.final_layer_dim = 7*7  # TODO correct?
        # first layer 
        self.conv1 = nn.Conv2d(input_channel, n1, filtersize, padding=padding)
        self.batchnorm = nn.BatchNorm2d(n1)   # is 2d the right one? ---> yes
        self.pool = nn.MaxPool2d(2)
        self.prelu = nn.PReLU()
        # second layer
        self.conv2 = nn.Conv2d(n1, n2, filtersize, padding=padding)
        self.batchnorm2 = nn.BatchNorm2d(n2) 
        self.prelu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(2)
        #third layer 
        self.conv3 = nn.Conv2d(n2, n3, filtersize, padding=padding)
        self.batchnorm3 = nn.BatchNorm2d(n3) 
        self.prelu3 = nn.PReLU()
        #self.pool3 = nn.MaxPool2d(2)
        
        self.fc1 = nn.Linear(n3 * self.final_layer_dim, 10)

    def forward(self, x):
        # first layer
        x = self.batchnorm(self.conv1(x))
        #print('batchnorm ', x.size())
        x = self.pool(self.prelu(x))
        #print('shape first layer ', x.size())
        
        # second layer
        x = self.batchnorm2(self.conv2(x))
        #print('after batchnorm 2 ', x.size())
        x = self.pool2(self.prelu2(x))
        #print('shape second layer ', x.size())
        
        # third layer
        x = self.batchnorm3(self.conv3(x))
        #print('after batchnorm 3 ', x.size())
        #x = self.pool3(self.prelu3(x))
        #print('shape third layer ', x.size())
        
        x = x.view(-1, self.n3 * self.final_layer_dim)
        #print('shape third layer after view', x.size())
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)
        #print('after softmax ', x.size())
        return x
    
        
    
    
