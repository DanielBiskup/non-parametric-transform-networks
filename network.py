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




""" Non-parametric transformation network """ # TODO make this a layer
class NPTN(nn.Module):
    def __init__(self):
        self.M=3
        self.N=2 
        self.G=5
        filtersize = 4
        
        super(NPTN, self).__init__()
        
        self.conv1 = nn.Conv2d(self.M, self.M*self.N*self.G, filtersize, groups=self.M) # in, out, kernel size, groups as in paper
        self.maxpool3d = nn.MaxPool3d((self.G, 1, 1)) 
        self.meanpool3d = nn.AvgPool3d((self.M, 1, 1)) # Is that the right pooling? - AvgPool3d?
                

    def forward(self, x):
        print('\nShape of x ', x.size())
        x = self.conv1(x)
        print('Shape after convolution', x.size())
        x = self.maxpool3d(x)
        print("Shape after MaxPool3d: ", x.size()) # dimension should be M*N
        permutation = torch.from_numpy(np.array([j for i in range(self.M) for j in range(self.N)]))
        print('permutation ', permutation)
        x = x[:, permutation] # reorder channels
        print("Shape after Channel reordering: ", x.size())
        x = self.meanpool3d(x)
        print('Shape after Mean Pooling: ', x.size())
        return x


net = NPTN()
outputs = net(Variable(images))

#print(outputs)



''' Testing area 
# test channel reordering
permutation = [2,1,0]
test_tensor = torch.cat([images,images,images])

imshow(torchvision.utils.make_grid(test_tensor))
#print(test_tensor)
print(test_tensor.size())
reordered_t = test_tensor[:,permutation] # doesn't work

#print(reordered_t)
imshow(torchvision.utils.make_grid(reordered_t))

print(reordered_t.size())
'''
