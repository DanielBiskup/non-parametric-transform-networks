# -*- coding: utf-8 -*-
"""
Cognitive Robotics Lab

Non-parameteric transformation network
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F




def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


print(torch.cuda.is_available())


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


# show a few images, taken from tutorial

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))



""" Non-parametric transformation network """
class NPTN(nn.Module):
    def __init__(self, M=3, N=2, G=3, filtersize):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(M, M*N*G, filtersize, groups=M) # in, out, kernel size, groups as in paper
        self.pool = nn.MaxPool3d((G, 1, 1)) # tuple needed? in right order?
        # check dimension (should be M*N)
        # channel reordering how?
        
        # Do mean pooling - AvgPool3d?
        
        # what now? make this a layer?
        
        # rest stuff copied from tutorial
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = NTPN()








