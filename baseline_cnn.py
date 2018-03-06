#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 09:33:35 2018

"""
print('Training the Baseline 2-layered CNN on CIFAR10')


import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from network import NPTN
import pandas as pd
import sys

###############   Test if you can use the GPU   ################

use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
    print('Using CUDA')

###########   loading and preprocessing the data    ############

# load dataset CIFAR10, normalize, crop and flip as in paper
transform = transforms.Compose(
    [transforms.RandomCrop(32, padding=4),
     transforms.RandomHorizontalFlip(), 
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



#############        Network definition       ####################

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

kernel_size = 7
file_name = 'cnn_ks=' + str(kernel_size) + '.csv'

netN24G2 = twoLayeredCNN(filtersize=kernel_size)
net = netN24G2

if use_cuda:
    net.cuda()

############## Chooses optimizer and loss  ##############

criterion = nn.NLLLoss()  #TODO which things here?!
optimizer = optim.SGD(net.parameters(), lr=0.1)


############## Train the network  ######################


stat_epoch = list()
stat_batch = list()
stat_loss = list()


num_epochs = 300 # paper: 300

# (taken from tutorial) 
for epoch in range(num_epochs):  # loop over the dataset multiple times

    if epoch == 150:
        optimizer = optim.SGD(net.parameters(), lr=0.01)
        print('Learning rate adapted') # TODO change learning rate (optimizer? after certain iterations)
    if epoch == 225:
        optimizer = optim.SGD(net.parameters(), lr=0.001)
        print('Learning rate adapted')
        
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0): 
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        if use_cuda:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:    
            inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 500 == 499:    
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 500)) 
            running_loss = 0.0

print('Finished Training')

# Save Data to CSV
stats_df = pd.DataFrame(
    {'epoch': stat_epoch,
     'batch': stat_batch,
     'loss': stat_loss
    })
    
    

################       Test the network        ##########################


# measure accuracy (not in paper though, so could be removed), currently not working
correct = 0
total = 0
running_loss = 0.0

for data in testloader:
    images, labels = data
    if use_cuda:
        images, labels = Variable(images.cuda()), Variable(labels.cuda())
    else:    
        images, labels = Variable(images), Variable(labels)

    outputs = net(images)
    loss = criterion(outputs, labels)
    
    
    running_loss += loss.data[0] * images.size(0)
    
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.data).sum()

print('Accuracy of the CNN network on the 10000 test images: %d %%' % (
    100 * correct / total))
print('Cross entropy loss = ', running_loss /testloader.dataset.test_data.shape[0])


stats_df.to_csv(file_name)