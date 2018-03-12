#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 16:30:20 2018

loads one batch of CIFAR images into variable batch for testing networks
"""

import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms

from three_layered_models import threeLayeredCNN, threeLayeredNPTN
from torch.nn.functional import affine_grid, grid_sample

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


#print(torch.cuda.is_available()) # is not and code can not use cuda (yet?)


# load dataset CIFAR10, if not available download and extract
# images are normalized to range [-1,1], taken from tutorial 
transform = transforms.Compose(
    [transforms.ToTensor(),
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

# load MNIST data and rotate it
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
    ])), batch_size=64, shuffle=True, num_workers=4)



# show a few images, taken from tutorial

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

data_iter = iter(train_loader)
images_mnist, labels_mnist = data_iter.next()

# show images
#imshow(torchvision.utils.make_grid(images))
# print labels
#print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

batch = Variable(images)


net_3_NPTN = threeLayeredNPTN()
net_3_CNN = threeLayeredCNN()


def make_rotation_matrix(rot_deg):
    rot_rad = np.radians(rot_deg)
    c = np.cos(rot_rad)
    s = np.sin(rot_rad)
    mat = np.array([[c,-s,0],[s,c,0]])
    return mat

a = make_rotation_matrix(90)
print(a)

def make_rotation_batch(rot_deg, batch_size):
    return np.array([make_rotation_matrix(rot_deg) for i in range(batch_size)])

b = make_rotation_batch(-90, 4)
print(b.shape)

im = images[:4]
#show(im)

flow_field = affine_grid(torch.Tensor(b), im.size())
whatever = grid_sample(images, flow_field)

show(whatever.data)


