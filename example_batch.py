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
import torch.nn.functional as F

from three_layered_models import threeLayeredCNN, threeLayeredNPTN
from torch.nn.functional import affine_grid, grid_sample
from filter_visualization import plot_kernels, example_plots_cnn, example_plots_nptn
from filter_visualization import imshow_cifar, imshow_mnist
from filter_visualization import show_cifar, show_mnist


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
    ])), batch_size=64, shuffle=False, num_workers=4)



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

#####      plot some filters from loaded network        ##########


nptn_net = torch.load('2018_03_12_13_49_53_NEWAccTEST_MNIST_rot_90_yaml.model')
example_plots_nptn(nptn_net, layer=1, num_samples=10)


########   playground rotating kernels and passing them to a convolution    #########

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

def make_rotations(rot_deg_min, rot_deg_max, num_rots):
    return np.array([make_rotation_matrix(rot_deg) for rot_deg in np.linspace(rot_deg_min, rot_deg_max, num_rots)])

# for some images
b = make_rotation_batch(-90, 4)
print(b.shape)

im = images[:4]
#show(im)

flow_field = affine_grid(torch.Tensor(b), im.size())
whatever = grid_sample(images, flow_field)

show_cifar(whatever.data)

# for kernels
net = torch.load('2018_03_12_13_49_53_NEWAccTEST_MNIST_rot_90_yaml.model')
cnn_net = torch.load( '2018_03_09_11_35_25_mnist_1M_cnn_2layers36N1_16N2_5Kernel.final_model')
t = net.nptn.conv1.weight.data.cpu()[:1]
kernel3d = cnn_net.conv2.weight[:1,:3].cpu()

# duplicate kernel
print(t.shape)
plot_kernels(t.numpy())

flow_field2 = affine_grid(torch.Tensor(make_rotation_batch(45,1)), t.size())
whatever2 = grid_sample(t, flow_field2)

plot_kernels(whatever2.data.numpy())


####   make nice example kernel    ########
k = np.concatenate((np.ones((2,5)),np.zeros((3,5))),0)
kn = 0.5*np.ones((5,5))
ki = np.concatenate((0.6*np.ones((3,5)),0.3*np.ones((2,5))),0)
k3d = np.array([k,ki,kn])
k = torch.unsqueeze(torch.unsqueeze(torch.Tensor(k),0),0)
k3d = torch.unsqueeze(torch.Tensor(k3d),0)
k = Variable(k)
k3d = Variable(k3d)


###### make ugly examply image  ######
pic1 = np.concatenate((np.zeros((12,32)),np.ones((8,32)),np.zeros((12,32))),0)
pic2 = 0.5 * np.ones((32,32))
pic = np.array([pic1,pic2,pic2])
pic =  torch.unsqueeze(torch.Tensor(pic),0)

# make several rotations from one kernel
kernel = k

def rotate(rot_mat, img):
    flow_field = affine_grid(torch.Tensor(rot_mat), img.size())
    return grid_sample(img, flow_field)


# for black and white images
g = 5
rot_mats = torch.Tensor(make_rotations(-180,180,g))
rotated_kernels = torch.cat([rotate(torch.unsqueeze(rotation,0), kernel) for rotation in rot_mats])
plot_kernels(rotated_kernels.data.numpy())

# how to pass convolution your own filters 
example_imgs = images_mnist[:5]
show_mnist(example_imgs)
resim = F.conv2d(Variable(example_imgs), rotated_kernels)
plot_kernels(resim.data.numpy(), num_cols=g)

# for RGB images (using 3d kernels)
kernel3d = k3d
g = 5
num_imgs = 1
cif_imgs = pic #images[:num_imgs]
show_cifar(cif_imgs)
plot_kernels(kernel3d.data.numpy())

rot_mats_3 = torch.Tensor(make_rotations(-180,180,g))
rotated_kernels_3 = torch.cat([rotate(torch.unsqueeze(rotation,0), kernel3d) for rotation in rot_mats_3])
right_shape = rotated_kernels_3.shape
#rotated_kernels_3 = rotated_kernels_3.permute(1,0,2,3) # reorders kernel such that the rotated kernels are in a row
#rotated_kernels_3 = rotated_kernels_3.contiguous().view(right_shape)
plot_kernels(rotated_kernels_3.data.numpy(), num_cols=3)

resim_cif = F.conv2d(Variable(cif_imgs), rotated_kernels_3)
plot_kernels(resim_cif.data.numpy(), num_cols=g)


############## TEST AREA ######################################################
''''
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
'''

'''
print(torch.cuda.is_available()) # is not and code can not use cuda (yet?)


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
'''