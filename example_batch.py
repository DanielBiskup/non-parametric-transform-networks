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


#nptn_net = torch.load('2018_03_12_13_49_53_NEWAccTEST_MNIST_rot_90_yaml.model')
#example_plots_nptn(nptn_net, layer=1, num_samples=10)


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
#net = torch.load('2018_03_12_13_49_53_NEWAccTEST_MNIST_rot_90_yaml.model')
#cnn_net = torch.load( '2018_03_09_11_35_25_mnist_1M_cnn_2layers36N1_16N2_5Kernel.final_model')
#t = net.nptn2.conv1.weight.data.cpu()
#kernel3d = cnn_net.conv2.weight[:1,:3].cpu()

# duplicate kernel
#print(t.shape)
#print(kernel3d.shape)
#plot_kernels(t.numpy())

#flow_field2 = affine_grid(torch.Tensor(make_rotation_batch(45,1)), t.size())
#whatever2 = grid_sample(t, flow_field2)

#plot_kernels(whatever2.data.numpy())


####   make nice example kernel    ########
k = np.concatenate((np.ones((2,5)),np.zeros((3,5))),0)
k[0,0] = 1
kn = 0.5*np.ones((5,5))
ki = np.concatenate((0.6*np.ones((5,3)),0.3*np.ones((5,2))),1)
k3d = np.array([kn,ki,k])
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
N=3
num_imgs = 1
cif_imgs = pic #images[:num_imgs]
show_cifar(cif_imgs)
plot_kernels(kernel3d.data.numpy())

rot_mats_3 = torch.Tensor(make_rotations(-180,180,g))
rotated_kernels_3 = torch.cat([rotate(torch.unsqueeze(rotation,0), kernel3d) for rotation in rot_mats_3])
#right_shape = rotated_kernels_3.shape
#rotated_kernels_3 = rotated_kernels_3.permute(1,0,2,3) # reorders kernel such that the rotated kernels are in a row
#rotated_kernels_3 = rotated_kernels_3.contiguous().view(right_shape)
plot_kernels(rotated_kernels_3.data.numpy(), num_cols=3)

rot_mats_more = torch.Tensor(make_rotations(-180,180,g))
rotated_kernels_more =  torch.cat([rotate(torch.unsqueeze(rotation,0), kernel) for rotation in rot_mats_more])
#resim_cif = F.conv2d(Variable(cif_imgs), rotated_kernels_3, groups=3)
resim_cif = F.conv2d(Variable(cif_imgs), rotated_kernels, groups=3)
plot_kernels(resim_cif.data.numpy(), num_cols=g)

############    New NPTN rotation test area    ##############

num_imgs=4
cif_imgs = images[:num_imgs]

####   make nice example kernel    ########
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

g = 5
N=1
M=3
#rot_mats = torch.Tensor(make_rotations(-180,180,g))
#rotated_kernels = torch.cat([rotate(torch.unsqueeze(rotation,0), kernel) for rotation in rot_mats])
#right_shape = rotated_kernels.shape

#rotated_kernels_3 = rotated_kernels_3.permute(1,0,2,3) # reorders kernel such that the rotated kernels are in a row
#rotated_kernels_3 = rotated_kernels_3.contiguous().view(right_shape)
#plot_kernels(rotated_kernels_3.data.numpy(), num_cols=3)

    
    
rot_mats_more = torch.Tensor(make_rotations(-80,80,g))
rotated_kernels_more =  torch.cat([rotate(torch.unsqueeze(rotation,0), k3d) for rotation in rot_mats_more])

#resim_cif = F.conv2d(Variable(cif_imgs), rotated_kernels_3, group
plot_kernels(rotated_kernels_more.data.numpy())
print(rotated_kernels_more.shape)

r_k_m = rotated_kernels_more.view(g,M*N,5,5)
plot_kernels(r_k_m.data.numpy(), num_cols=5)
r_k_m = torch.transpose(r_k_m, 0,1)
plot_kernels(r_k_m.data.numpy(), num_cols=5)
#print(rotated_kernels_more.shape)
r_k_m_flat = r_k_m.contiguous().view(g*M*N,1,5,5)

plot_kernels(r_k_m_flat.data.numpy(), num_cols=5)
print(r_k_m_flat.shape)

resim_cif = F.conv2d(Variable(cif_imgs), r_k_m_flat, groups=3)
plot_kernels(resim_cif.data.numpy(), num_cols=g)




def get_rotated_kernels(kernels, G, rot_max, rot_min,):
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
    
# number of kernels passed == number of channels leaving
def forward_pass(imgs, kernels=kd3, M=3, G=4, rot_max=-90, rot_min=90, plot=False):
    rot_k = get_rotated_kernels(kernels, G, rot_max, rot_min)   
    convoluted_imgs = F.conv2d(Variable(imgs), rot_k, groups=M)
    
    if plot:
        plot_kernels(rot_k.data.numpy(), num_cols=G)
        plot_kernels(convoluted_imgs.data.numpy(), num_cols=G)
    return convoluted_imgs

conv_imgs = forward_pass(images_mnist[:4], kernels=k, M=1, G=5, plot=True)

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