# -*- coding: utf-8 -*-
"""
Cognitive Robotics Lab

Non-parameteric transformation network
"""
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

def make_permutation(M,N):
    nums = [i for i in range(M*N)]
    NUMS = np.array(nums)
    NUMS = np.reshape(NUMS, (M,N))
    NUMS = NUMS.T
    return NUMS.flatten()


""" Non-parametric transformation network layer """
class NPTN(nn.Module):
    def __init__(self, M, N, G, filtersize, padding=0):
        super(NPTN, self).__init__()
        self.M=M
        self.N=N 
        self.G=G
        
        self.conv1 = nn.Conv2d(self.M, self.M*self.N*self.G, filtersize, groups=self.M, padding=padding) # in, out, kernel size, groups as in paper
        self.maxpool3d = nn.MaxPool3d((self.G, 1, 1))
        self.meanpool3d = nn.AvgPool3d((self.M, 1, 1)) # Is that the right pooling? - AvgPool3d?
        
        self.permutation = make_permutation(self.M, self.N)

    def forward(self, x):
        #print('\nShape of x ', x.size())
        x = self.conv1(x)
        #print('Shape after convolution', x.size())
        x = self.maxpool3d(x)
        #print("Shape after MaxPool3d: ", x.size()) # dimension should be M*N
        
        #print('permutation ', permutation)
        x = x[:, self.permutation] # reorder channels
        #print("Shape after Channel reordering: ", x.size())
        x = self.meanpool3d(x)
        #print('Shape after Mean Pooling: ', x.size())
        return x

class twoLayeredNPTN(nn.Module):
    def __init__(self, N, G, filtersize, in_channels=3):
        super(twoLayeredNPTN, self).__init__()
        if in_channels == 3:   # CIFAR
            self.final_layer_dim = (7-np.int(filtersize/1.7))**2 # works for filtersizes 3,5,7
        else: # MNIST
            self.final_layer_dim = 4 * 4
        # first layer 
        self.N = N
        self.nptn = NPTN(in_channels, N, G, filtersize)
        self.batchnorm = nn.BatchNorm2d(N)   # is 2d the right one?
        self.pool = nn.MaxPool2d(2)
        self.prelu = nn.PReLU()
        # second layer
        self.nptn2 = NPTN(N, 16, G, filtersize)
        self.batchnorm2 = nn.BatchNorm2d(16) 
        self.prelu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * self.final_layer_dim, 10) # TODO

    def forward(self, x):
        #print('============================================================')
        #print('the input x to the network ', x.size())
        x = self.nptn(x)
        #print('x after nptn ', x.size())
        x = self.batchnorm(x)
        #print('batchnorm ', x.size())
        #x = F.prelu(self.nptn(x), 0.1) # TODO: What about this?
        x = self.pool(self.prelu(x))
        #print('shape first layer ', x.size())
        x = self.nptn2(x)
        #print('x after nptn2 ', x.size())
        x = self.batchnorm2(x)
        #print('after batchnorm 2 ', x.size())
        x = self.pool2(self.prelu2(x))
        #print('shape second layer ', x.size())
        
        x = x.view(-1, 16 * 4 * 4) # TODO
        #print('shape second layer after flattening', x.size())
        
        # BUG DESCRIPTION:
        # Runtime Error:"invalid argument 2: size '[-1 x 400]' is invalid for input with 16384 elements"
        # when calling experiment with MNIST_rot_60.yaml
        # Maybe find information here: http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py
        
        x = self.fc1(x)
        #print('after Linear layer ', x.size()) 
        x = F.log_softmax(x, dim=1)
        #print('after softmax ', x.size())
        return x
    
    def num_flat_features(self, x):
        #copied from: http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
class twoLayeredCNN(nn.Module):
    def __init__(self, filtersize, in_channels=3, N1=48, N2=16):
        super(twoLayeredCNN, self).__init__()
        if in_channels==3: # CIFAR
            self.input_size=(3,32,32)
        else:
            self.input_size=(1,28,28)
        
        self.N2 = N2
        # first layer 
        self.conv1 = nn.Conv2d(in_channels, N1, filtersize) # TODO maybe change filter size
        self.batchnorm = nn.BatchNorm2d(N1)   # is 2d the right one?
        self.pool = nn.MaxPool2d(2)
        self.prelu = nn.PReLU()
        # second layer
        self.conv2 = nn.Conv2d(N1, N2, filtersize)
        self.batchnorm2 = nn.BatchNorm2d(N2) 
        self.prelu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(2)
         
        n = self.num_flat_features(self.input_size)
        #print('num_flat_features = ' + str(n))
        
        self.fc1 = nn.Linear(n, 10)
        
    def num_flat_features(self, input_size):
        t = Variable(torch.ones(1, *input_size))
        #print('t.size() = ' + str(t.size()))
        f = self.features(t)
        #print('Shape after convolution layers = ' + str(f.size()))
        n = int(np.prod(f.size()[1:]))
        return n

    def features(self, x):
        #print('input shape', x.shape)
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
        return x
        
    def forward(self, x):
        x = self.features(x)
        #x = x.view(-1, self.n3 * self.final_layer_dim)
        x = x.view(x.size(0), -1)
        #print('shape second layer ', x.size())
        x = F.log_softmax(self.fc1(x), dim=1)
        #print('after softmax ', x.size())
        return x

class threeLayeredNPTN(nn.Module):
    def __init__(self, filtersize=5, G=3, n1=48, n2=48, n3=16, input_channel=3):
        super(threeLayeredNPTN, self).__init__()
        self.n3 = n3
        padding = int(filtersize/2) # needed if you want to use maxpooling 3 times
        if input_channel==3: # CIFAR
            self.input_size=(3,32,32)
        else:
            self.input_size=(1,28,28)
        
        # first layer 
        self.nptn = NPTN(input_channel, n1, G, filtersize, padding=padding)
        self.batchnorm = nn.BatchNorm2d(n1)   # is 2d the right one?
        self.pool = nn.MaxPool2d(2)
        self.prelu = nn.PReLU()
        # second layer
        self.nptn2 = NPTN(n1, n2, G, filtersize, padding=padding)
        self.batchnorm2 = nn.BatchNorm2d(n2) 
        self.prelu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(2)
        #third layer 
        self.nptn3 = NPTN(n2, n3, G, filtersize, padding=padding)
        self.batchnorm3 = nn.BatchNorm2d(n3) 
        self.prelu3 = nn.PReLU()
        #self.pool3 = nn.MaxPool2d(2)
        
        n = self.num_flat_features(self.input_size)
        #print('num_flat_features = ' + str(n))
        
        self.fc1 = nn.Linear(n, 10)

    def num_flat_features(self, input_size):
        t = Variable(torch.ones(1, *input_size))
        #print('t.size() = ' + str(t.size()))
        f = self.features(t)
        #print('Shape after convolution layers = ' + str(f.size()))
        n = int(np.prod(f.size()[1:]))
        return n

    def features(self, x):
        #print('x.size() = ' + str(x.size()))
        
        # first layer
        x = self.batchnorm(self.nptn(x))
        #print('batchnorm ', x.size())
        x = self.pool(self.prelu(x))
        #print('shape first layer ', x.size())
        
        # second layer
        x = self.batchnorm2(self.nptn2(x))
        #print('after batchnorm 2 ', x.size())
        x = self.pool2(self.prelu2(x))
        #print('shape second layer ', x.size())
        
        # third layer
        x = self.batchnorm3(self.nptn3(x))
        #print('after batchnorm 3 ', x.size())
        x = self.prelu3(x)
        #print('shape third layer ', x.size())
        return x
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.log_softmax(self.fc1(x), dim=1)
        return x

class threeLayeredCNN(nn.Module):
    def __init__(self, filtersize=5, n1=89, n2=89, n3=16, input_channel=3):
        super(threeLayeredCNN, self).__init__()
        self.n3 = n3
        padding = int(filtersize/2) # needed if you want to use maxpooling 3 times
        if input_channel==3: # CIFAR
            self.input_size=(3,32,32)
        else:
            self.input_size=(1,28,28)
            
        # first layer 
        self.conv1 = nn.Conv2d(input_channel, n1, filtersize, padding=padding)
        self.batchnorm = nn.BatchNorm2d(n1)   # is 2d the right one?
        self.pool = nn.MaxPool2d(2)
        self.prelu = nn.PReLU()
        # second layer
        self.conv2 = nn.Conv2d(n1, n2, filtersize, padding=padding)
        self.batchnorm2 = nn.BatchNorm2d(n2) 
        self.prelu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(2)
        #third layer 
        self.conv3 = nn.Conv2d(n2, n3, filtersize, padding=padding)
        self.batchnorm3 = nn.BatchNorm2d(n3)
        self.prelu3 = nn.PReLU()
        #self.pool3 = nn.MaxPool2d(2)
        
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
        x = self.batchnorm(self.conv1(x))
        #print('batchnorm ', x.size())
        x = self.pool(self.prelu(x))
        #print('shape first layer ', x.size())
        
        # second layer
        x = self.batchnorm2(self.conv2(x))
        #print('after batchnorm 2 ', x.size())
        x = self.pool2(self.prelu2(x))
        #print('shape second layer ', x.size())
        
        # third layer
        #x = self.batchnorm3(self.conv3(x))
        #print('after batchnorm 3 ', x.size())
        #x = self.prelu3(x)
        #print('shape third layer ', x.size())
        return x
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.log_softmax(self.fc1(x), dim=1)
        return x
<<<<<<< HEAD
    
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
=======
   
####   from here on not functional stuff !!       #####
        
def rotate(rot_mat, img):
    flow_field = affine_grid(torch.Tensor(rot_mat), img.size())
    return grid_sample(img, flow_field)
    
    
class rotConv(nn.Module):
    def __init__(self, M, N, G, filtersize, rot_min=-180, rot_max=180, padding=0):

        super(rotConv, self).__init__()
        self.M=M
        self.N=N 
        self.G=G
        
        rot_mats = torch.Tensor(make_rotations(rot_min,rot_max,self.G))
        rotated_kernels = torch.cat([rotate(torch.unsqueeze(rotation,0), kernel) for rotation in rot_mats])

        #self.conv1 = nn.Conv2d(self.M, self.M*self.N*self.G, filtersize, groups=self.M, padding=padding) # in, out, kernel size, groups as in paper
        #self.maxpool3d = nn.MaxPool3d((self.G, 1, 1)) 
        #self.meanpool3d = nn.AvgPool3d((self.M, 1, 1)) # Is that the right pooling? - AvgPool3d?
        
        #self.permutation = make_permutation(self.M, self.N)

    def forward(self, x):
        #print('\nShape of x ', x.size())
        x = self.conv1(x)
        #print('Shape after convolution', x.size())
        x = self.maxpool3d(x)
        #print("Shape after MaxPool3d: ", x.size()) # dimension should be M*N
        
        #print('permutation ', permutation)
        x = x[:, self.permutation] # reorder channels
        #print("Shape after Channel reordering: ", x.size())
        x = self.meanpool3d(x)
        #print('Shape after Mean Pooling: ', x.size())
        return x    
    
    
'''  
    
class rotNet(nn.Module):
    def __init__(self, filtersize=5, G=4 , n1=9, n2=16, input_channel=3):
        super(rotNet, self).__init__()
>>>>>>> 25a66e08cc8017e487eda0e7de7d7689ffa21183

        if input_channel==3: # CIFAR
            self.input_size=(3,32,32)
        else:
            self.input_size=(1,28,28)
        #self.n3 = n3
        padding = int(filtersize/2) # do or don't ?
        
        # first layer 
        self.conv1 = nn.Conv2d(input_channel, n1, filtersize, padding=padding)
        self.batchnorm = nn.BatchNorm2d(n1)   # is 2d the right one?
        self.pool = nn.MaxPool2d(2)
        self.prelu = nn.PReLU()
        # second layer
        self.conv2 = nn.Conv2d(n1, n2, filtersize, padding=padding)
        self.batchnorm2 = nn.BatchNorm2d(n2) 
        self.prelu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(2)
        
        n = self.num_flat_features(self.input_size)
        
        self.fc1 = nn.Linear(n, 10)
        
    def features(self, x):
        # first layer
        #x = self.batchnorm(self.conv1(x))
        #print('batchnorm ', x.size())
        x = self.pool(self.prelu(x))
        #print('shape first layer ', x.size())
        
        # second layer
        x = self.batchnorm2(self.conv2(x))
        #print('after batchnorm 2 ', x.size())
        x = self.pool2(self.prelu2(x))
        #print('shape second layer ', x.size())
        return x
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.log_softmax(self.fc1(x), dim=1)
        return x

# show images
# imshow(torchvision.utils.make_grid(images))
# print labels
<<<<<<< HEAD
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
'''


# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

