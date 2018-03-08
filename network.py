# -*- coding: utf-8 -*-
"""
Cognitive Robotics Lab

Non-parameteric transformation network
"""
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def make_permutation(M,N):
    nums = [i for i in range(M*N)]
    NUMS = np.array(nums)
    NUMS = np.reshape(NUMS, (M,N))
    NUMS = NUMS.T
    return NUMS.flatten()


""" Non-parametric transformation network layer """
class NPTN(nn.Module):
    def __init__(self, M, N, G, filtersize, padding=0):
        self.M=M
        self.N=N 
        self.G=G
        
        super(NPTN, self).__init__()
        
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
        self.final_layer_dim = (7-np.int(filtersize/1.7))**2   # works for filtersizes 3,5,7
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
        self.fc1 = nn.Linear(16 * self.final_layer_dim, 10)

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
        
        x = x.view(-1, 16 * self.final_layer_dim)
        #print('shape second layer ', x.size())
        x = F.log_softmax(self.fc1(x), dim=1)
        #print('after softmax ', x.size())
        return x
    
class twoLayeredCNN(nn.Module):
    def __init__(self, filtersize, in_channels=3, N1=48, N2=16):
        super(twoLayeredCNN, self).__init__()
        self.final_layer_dim = (7-np.int(filtersize/1.7))**2   # works for filtersizes 3,5,7
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
         
        self.fc1 = nn.Linear(16 * self.final_layer_dim, 10)

    def forward(self, x):
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
        
        x = x.view(-1, 16 * self.final_layer_dim)
        #print('shape second layer ', x.size())
        x = F.log_softmax(self.fc1(x), dim=1)
        #print('after softmax ', x.size())
        return x

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