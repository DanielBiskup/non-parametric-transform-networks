#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 09:20:49 2018

first experiment

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
from network import NPTN


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

class twoLayeredNPTN(nn.Module):
    def __init__(self, N, G):
        super(twoLayeredNPTN, self).__init__()
        #self.weight =  # torch tensor, but which size?
        # first layer 
        self.N = N
        self.nptn = NPTN(3, N, G, 3) # TODO maybe change filter size
        self.batchnorm = nn.BatchNorm2d(N)   # is 2d the right one?
        self.pool = nn.MaxPool2d(2)
        self.prelu = nn.PReLU()
        # second layer
        self.nptn2 = NPTN(N, N, G, 3)
        self.prelu2 = nn.PReLU()
         
        self.fc1 = nn.Softmax(N * 6 * 6, 10)

    def forward(self, x):
        x = self.nptn(x)
        #print('x after nptn ', x.size())
        x = self.batchnorm(x)
        #print('batchnorm ', x.size())
        #x = F.prelu(self.nptn(x), 0.1) 
        x = self.pool(self.prelu(x))
        #print('shape first layer ', x.size())
        x = self.batchnorm(self.nptn2(x))
        #print('after batchnorm 2 ', x.size())
        x = self.pool(self.prelu2(x))
        #print('shape second layer ', x.size())
        
        x = x.view(-1, self.N * 6 * 6)

        x = self.fc1(x)
        return x


netN24G2 = twoLayeredNPTN(24,2)
net = netN24G2

if use_cuda:
    net.cuda()

############## Chooses optimizer and loss  ##############

criterion = nn.CrossEntropyLoss()   #TODO which things here?!
optimizer = optim.SGD(net.parameters(), lr=0.1)


############## Train the network  ######################

num_epochs = 2 # paper: 300

# (taken from tutorial) 
for epoch in range(num_epochs):  # loop over the dataset multiple times

    if epoch == 150:
        optimizer = optim.SGD(net.parameters(), lr=0.09)
        print('Learning rate adapted') # TODO change learning rate (optimizer? after certain iterations)
    if epoch == 225:
        optimizer = optim.SGD(net.parameters(), lr=0.08)
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

################       Test the network        ##########################


# measure accuracy (not in paper though, so could be removed)
correct = 0
total = 0
running_loss = 0.0

for data in testloader:
    images, labels = data
    if use_cuda:
        images, labels = Variable(images.cuda()), Variable(labels.cuda())
    else:    
        images, labels = Variable(images), Variable(labels)

    outputs = net(Variable(images))
    loss = criterion(outputs, labels)
    
    
    running_loss += loss.data[0] * images.size(0)
    
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
print('Ćross entropy loss = ', running_loss /testloader.dataset.test_data.shape[0])
