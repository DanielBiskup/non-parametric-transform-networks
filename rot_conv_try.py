#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 15:32:37 2018

@author: lab
"""

import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F

from three_layered_models import threeLayeredCNN, threeLayeredNPTN
from torch.nn.functional import affine_grid, grid_sample
from filter_visualization import plot_kernels, example_plots_cnn, example_plots_nptn
from filter_visualization import imshow_cifar, imshow_mnist
from filter_visualization import show_cifar, show_mnist
import torch.nn as nn
from torch.nn.parameter import Parameter
from network import make_permutation
import torch.optim as optim


def make_rotation_matrix(rot_deg):
    rot_rad = np.radians(rot_deg)
    c = np.cos(rot_rad)
    s = np.sin(rot_rad)
    mat = np.array([[c,-s,0],[s,c,0]])
    return mat

def make_rotations(rot_deg_min, rot_deg_max, num_rots):
    return np.array([make_rotation_matrix(rot_deg) for rot_deg in np.linspace(rot_deg_min, rot_deg_max, num_rots)])

def rotate(rot_mat, img):
    flow_field = affine_grid(torch.Tensor(rot_mat), img.size())
    return grid_sample(img, flow_field)


def get_rotated_kernels(kernels, G, rot_min, rot_max):
    num_kernels = kernels.shape[0]
    kernel_size = kernels.shape[-1]
    
    # make rotation matrices
    rot_mats = torch.Tensor(make_rotations(rot_min,rot_max,G))
    
    # rotate kernels
    kernelsPM = kernels.permute(1,0,2,3) # from (N,1,ks,ks) to (1,N,ks,ks)
    rot_kernelsPM = torch.cat([rotate(torch.unsqueeze(rotation,0), kernelsPM) for rotation in rot_mats])
    
    # sort kernels in appropiate order
    rot_kernels = rot_kernelsPM.view(G, num_kernels, kernel_size, kernel_size)
    rot_kernels = torch.transpose(rot_kernels, 0, 1)
    rot_kernels = rot_kernels.contiguous().view(G*num_kernels, 1, kernel_size, kernel_size)
    
    return rot_kernels

# rotates kernel and convolves images with kernels    
# number of kernels passed == number of channels leaving
    
def rotConv(imgs, kernels, M=3, G=4, rot_min=-90,  rot_max=90, plot=False):
    rot_k = get_rotated_kernels(kernels, G, rot_max, rot_min)   
    convoluted_imgs = F.conv2d(Variable(imgs), rot_k, groups=M)
    
    if plot:
        plot_kernels(rot_k.data.numpy(), num_cols=G)
        plot_kernels(convoluted_imgs.data.numpy(), num_cols=G)
    return convoluted_imgs


def make_rotConv(kernels, M=3, G=4, rot_min=-90,  rot_max=90, plot=False):      
    def rotConv(imgs):
        rot_k = get_rotated_kernels(kernels, G, rot_max, rot_min)   
        convoluted_imgs = F.conv2d(imgs, rot_k, groups=M)
        
        if plot == all:
            plot_kernels(rot_k.data.numpy(), num_cols=G)
            plot_kernels(convoluted_imgs.data.numpy(), num_cols=G)
        return convoluted_imgs
    return rotConv
#conv_imgs = rotconv(images_mnist[:4], kernels=k, M=1, G=5, plot=True)
  
class rotConvLayer(nn.Module):
    def __init__(self, M, N, G, filtersize, rot_min=-90, rot_max=90, padding=0, plot=False):
        super(rotConvLayer, self).__init__()
        self.M=M
        self.N=N 
        self.G=G
        self.rot_min = rot_min
        self.rot_max = rot_max
        self.plot = plot
        self.k = filtersize
        
        self.w = Parameter(torch.randn(M*N, 1, self.k, self.k))
        self.rotConv = make_rotConv(self.w, M=M, G=G, rot_min=rot_min, rot_max=rot_max, plot=plot)


    def forward(self, x):
        #print('\nShape of x ', x.size())
        x = self.rotConv(x)
        if self.plot:
            plot_kernels(self.w)
        
        return x
    

class rotPTN(nn.Module):
    def __init__(self, M, N, G, filtersize, rot_min=-90, rot_max=90, padding=0, plot=False):
        super(rotPTN, self).__init__()
        self.M=M
        self.N=N 
        self.G=G
        
        self.rotConv = rotConvLayer(M, N, G, filtersize, rot_min=-90, rot_max=90, padding=padding, plot=plot)
        self.maxpool3d = nn.MaxPool3d((self.G, 1, 1))
        self.meanpool3d = nn.AvgPool3d((self.M, 1, 1)) # Is that the right pooling? - AvgPool3d?
        
        self.permutation = make_permutation(self.M, self.N)

    def forward(self, x):
        #print('\nShape of x ', x.size())
        x = self.rotConv(x)
        #print('Shape after rot convolution', x.size())
        x = self.maxpool3d(x)
        #print("Shape after MaxPool3d: ", x.size()) # dimension should be M*N
        
        #print('permutation ', permutation)
        x = x[:, self.permutation] # reorder channels
        #print("Shape after Channel reordering: ", x.size())
        x = self.meanpool3d(x)
        #print('Shape after Mean Pooling: ', x.size())       
        return x


### does not work yet #### 
class rotNet(nn.Module):
    def __init__(self, input_channel=3, n1=9, n2=16, G=4, filtersize=5, rot_min=-90, rot_max=90, padding=0, plot=False):
        super(rotNet, self).__init__()

        self.M=input_channel
        self.n1=n1
        self.n2=n2
        self.G=G
        self.rot_min = rot_min
        self.rot_max = rot_max
        #self.plot = plot
        self.k = filtersize

        if input_channel==3: # CIFAR
            self.input_size=(3,32,32)
        else:
            self.input_size=(1,28,28)

        #padding = int(filtersize/2) # do or don't ?
        
        # first layer 
        self.rotPTN = rotPTN(input_channel, n1, G, filtersize, rot_min=-90, rot_max=90, padding=padding, plot=plot)
        #self.batchnorm = nn.BatchNorm2d(n1)   # is 2d the right one?
        self.pool = nn.MaxPool2d(2)
        self.prelu = nn.PReLU()
        # second layer
        #self.conv2 = nn.Conv2d(n1, n2, filtersize, padding=padding)
        #self.batchnorm2 = nn.BatchNorm2d(n2) 
        #self.prelu2 = nn.PReLU()
        #self.pool2 = nn.MaxPool2d(2)
        
        n = self.num_flat_features(self.input_size)
        
        self.fc1 = nn.Linear(n, 10)
        
    def num_flat_features(self, input_size):
        t = Variable(torch.ones(1, *input_size))
        #print('t.size() = ' + str(t.size()))
        f = self.features(t)
        #print('Shape after convolution layers = ' + str(f.size()))
        n = int(np.prod(f.size()[1:]))
        return n  
    
    def features(self, x):
        # first layer
        x = self.rotPTN(x)
        # x = self.batchnorm(x)
        #print('rotPTN ', x.size())
        x = self.pool(self.prelu(x))
        #print('shape first layer ', x.size())
        
        # second layer
        #x = self.batchnorm2(self.conv2(x))
        #print('after batchnorm 2 ', x.size())
        #x = self.pool2(self.prelu2(x))
        #print('shape second layer ', x.size())
        return x
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.log_softmax(self.fc1(x), dim=1)
        #print('shape prediction',x.shape)
        return x
    
    
#####  small run to see if weights are updated  #####

use_cuda = False

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



net = rotNet(1,2,4,5, plot=False)   
init_w = net.rotPTN.rotConv.w
print(net.rotPTN.rotConv.w)   
plot_kernels(net.rotPTN.rotConv.w, title='initialization')

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
            stat_loss = running_loss / 25
            print('[%d, %5d] loss: %.3f' %
                  (stat_epoch, stat_batch, stat_loss))
          
            running_loss = 0.0
            
    accuracy = (100 * correct / trainloader.dataset.train_data.shape[0])

    print('----------------------------------------------')
    print('Epoch ', epoch) 
    print('Accuracy of the network on the train images: %d %%' % accuracy)
    plot_kernels(net.rotPTN.rotConv.w)

num_epochs = 3 # paper: 300
for epoch in range(num_epochs):  # loop over the dataset multiple times

    training_epoch(epoch)
    
end_w = net.rotPTN.rotConv.w


