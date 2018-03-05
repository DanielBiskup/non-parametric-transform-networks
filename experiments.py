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
import pandas as pd


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
    def __init__(self, N, G, filtersize):
        super(twoLayeredNPTN, self).__init__()
        self.final_layer_dim = (7-np.int(filtersize/1.7))**2   # works for filtersizes 3,5,7
        # first layer 
        self.N = N
        self.nptn = NPTN(3, N, G, filtersize)
        self.batchnorm = nn.BatchNorm2d(N)   # is 2d the right one?
        self.pool = nn.MaxPool2d(2)
        self.prelu = nn.PReLU()
        # second layer
        self.nptn2 = NPTN(N, N, G, filtersize)
        self.batchnorm2 = nn.BatchNorm2d(N) 
        self.prelu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(2)
         
        self.fc1 = nn.Linear(N * self.final_layer_dim, 10)

    def forward(self, x):
        x = self.nptn(x)
        #print('x after nptn ', x.size())
        x = self.batchnorm(x)
        #print('batchnorm ', x.size())
        #x = F.prelu(self.nptn(x), 0.1) 
        x = self.pool(self.prelu(x))
        #print('shape first layer ', x.size())
        x = self.batchnorm2(self.nptn2(x))
        #print('after batchnorm 2 ', x.size())
        x = self.pool2(self.prelu2(x))
        #print('shape second layer ', x.size())
        
        x = x.view(-1, self.N * self.final_layer_dim)
        #print('shape second layer ', x.size())
        x = F.log_softmax(self.fc1(x), dim=1)
        #print('after softmax ', x.size())
        return x


netN24G2 = twoLayeredNPTN(48,1,7)
net = netN24G2

if use_cuda:
    net.cuda()

############## Chooses optimizer and loss  ##############

criterion = nn.NLLLoss()   #TODO which things here?!
optimizer = optim.SGD(net.parameters(), lr=0.1)


############## Train the network  ######################


num_epochs = 2 # paper: 300

stat_epoch = list()
stat_batch = list()
stat_loss = list()

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
            print('Start')
            stat_epoch.append(epoch + 1)
            stat_batch.append(i + 1)
            stat_loss.append(running_loss / 500)
            print('[%d, %5d] loss: %.3f' %
                  (stat_epoch[-1], stat_batch[-1], stat_loss[-1]))
            running_loss = 0.0
            print('Stop')

print('Finished Training')

# Save Data to CSV
stats_df = pd.DataFrame(
    {'epoch': stat_epoch,
     'batch': stat_batch,
     'loss': stat_loss
    })
    
stats_df.to_csv("stats.csv")

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

print('Accuracy of the NPTN network on the 10000 test images: %d %%' % (
    100 * correct / total))
print('Cross entropy loss = ', running_loss /testloader.dataset.test_data.shape[0])
