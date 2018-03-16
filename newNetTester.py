#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 09:40:28 2018

@author: lab
"""
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
#from rot_conv_try import get_rotated_kernels
import torchvision
from torchvision import datasets, transforms
from network import make_permutation
import torch.optim as optim
from filter_visualization import plot_kernels
from newNet import twoLayeredROTNET
# %% Testing
'''
# TEST KERNELS
k = np.concatenate((np.ones((2,5)),np.zeros((3,5))),0)
kn = 0.5*np.ones((5,5))
kn[2,2] = 1
ki = np.concatenate((0.6*np.ones((5,3)),0.3*np.ones((5,2))),1)
kl = np.concatenate((np.ones((5,1)),np.zeros((5,1)),np.ones((5,1)),np.zeros((5,1)),np.ones((5,1))),1)
kh = np.concatenate((np.ones((5,1)),np.zeros((5,3)),np.ones((5,1))),1)

k3d = np.array([kn,ki,k])
k6d = np.array([kl,kn,ki,kn,k,kh])

k = torch.unsqueeze(torch.unsqueeze(torch.Tensor(k),0),0)
k3d = torch.unsqueeze(torch.Tensor(k3d),0)
k6d = torch.unsqueeze(torch.Tensor(k6d),0)
k = Variable(k)
k3d = Variable(k3d)
k6d = Variable(k6d)

kd3 = k3d.permute(1,0,2,3)
kd6 = k6d.permute(1,0,2,3)
plot_kernels(kd3.data.numpy())
plot_kernels(kd6.data.numpy())

# NET CONF
M = 1 # M: number of input channeles
N = 6 # N: number of output channels
G = 3 # G: number of translations to use.
alpha = 90 
k = 5 # filtersize

rC = RTN_CORE(M, N, G, alpha = alpha, filtersize = k, padding=0, init_kernels=kd6)

# LOAD IMAGES
max_rotation = 60 
# Training dataset
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=True, download=True,
                   transform=transforms.Compose([
                       #torchvision.transforms.RandomCrop((32,32), padding=2),    
                       torchvision.transforms.RandomRotation(max_rotation),
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=64, shuffle=True, num_workers=4)
# Test dataset
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=False, transform=transforms.Compose([
        torchvision.transforms.RandomRotation(max_rotation),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=64, shuffle=False, num_workers=4)



# show a few images, taken from tutorial

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

data_iter = iter(train_loader)
images_mnist, labels_mnist = data_iter.next()

inputs = Variable(images_mnist) # size(g) = ( G x 2 x 3 ) # TODO
outputs = rC(inputs)
'''

# %% 
#####  small run to see if weights are updated  #####
use_cuda = True
plot = True

# load MNIST data and rotate it
max_rotation = 0
batch_size = 32
# Training dataset
trainloader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=True, download=True,
                   transform=transforms.Compose([
                       #torchvision.transforms.RandomCrop((32,32), padding=2),    
                       torchvision.transforms.RandomRotation(max_rotation),
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=batch_size, shuffle=True, num_workers=4)
# Test dataset
testloader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=False, transform=transforms.Compose([
        torchvision.transforms.RandomRotation(max_rotation),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=batch_size, shuffle=False, num_workers=4)


# DEV_VALUES__
M = 1 # M: number of input channeles
N = 2 # N: number of output channels
G = 5 # G: number of translations to use.
alpha = 5
k = 3 # filtersize
# inputs = Variable(torch.randn(G,2,3)) # size(g) = ( G x 2 x 3 ) # TODO
#__DEV_VALUES

# %%
#net = RTN_CORE(M, N, G, alpha = alpha, filtersize = k, padding=0)
#net = TestNet( M, N, G, alpha, filtersize = k, padding=0)
net = twoLayeredROTNET(M=M, N1=N, N2=3)

if use_cuda:
    net.cuda()
#init_w = net.rotPTN.rotConv.w.detach() # does not work since copy is not possible
#print(net.rotPTN.rotConv.w)   

if plot:
    #print(net.rtn_layer_1.rot_core.w)
    plot_kernels(net.rtn_layer_1.rot_core.w, title='initialization, layer 1')
    #plot_kernels(net.rotPTN2.rotConv.w, title='initialization, layer 2')
criterion = nn.NLLLoss() 
optimizer = optim.SGD(net.parameters(), lr=0.05)



def training_epoch(epoch):
    running_loss = 0.0
    correct = 0
    
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
        
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels.data).sum()
        
        if i % 250 == 249:
            stat_epoch = epoch + 1
            stat_batch = i + 1
            stat_loss = running_loss / 250
            print('[%d, %5d] loss: %.3f' %
                  (stat_epoch, stat_batch, stat_loss))
          
            running_loss = 0.0
            
    accuracy = (100 * correct / trainloader.dataset.train_data.shape[0])

    print('----------------------------------------------')
    print('Epoch ', epoch) 
    print('Accuracy of the network on the train images: %d %%' % accuracy)
    if plot:
        plot_kernels(net.rtn_layer_1.rot_core.w, title='layer 1')
        #plot_kernels(net.rotPTN2.rotConv.w, title='layer 2')

num_epochs = 3 # paper: 300
for epoch in range(num_epochs):  # loop over the dataset multiple times

    training_epoch(epoch)
    



