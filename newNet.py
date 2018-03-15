# http://pytorch.org/docs/0.3.1/nn.html#id15

template # the filters we actually want to learn.
# Expose them as parameters:  
# http://pytorch.org/docs/0.3.1/nn.html?#parameters

filters = autograd.Variable(torch.randn(8,4,3,3)) # calculate as transforms from "template"
inputs = autograd.Variable(torch.randn(1,4,5,5))

# Use the functional (torch.nn.functional.conv2d) instead of Module
# (torch.nn.conv2d), becuse we don't want trainable parameters exept the
# template vector defined above:
out_conv = F.conv2d(inputs, filters, padding=1)

# apply the same max_pooling we did for NPTNs to out_conv
# http://pytorch.org/docs/0.3.1/nn.html?#torch.nn.MaxPool3d

################################################

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from rot_conv_try import get_rotated_kernels


def make_permutation(M,N):
    nums = [i for i in range(M*N)]
    NUMS = np.array(nums)
    NUMS = np.reshape(NUMS, (M,N))
    NUMS = NUMS.T
    return NUMS.flatten()

M = 3 # M: number of input channeles
N = 2 # N: number of output channels
G = 5 # G: number of translations to use.
alpaha = 5
k = 3 # filtersize

class NewNPTN(nn.Module):
    def __init__(self, M, N, G, alpha, filtersize = 5, padding=0):
        super(NewNPTN, self).__init__()
        
        # M: number of input channeles
        # N: number of output channels
        # G: number of translations to use. If odd, the original filter will also be used. If even, not.
        # alpha: There will be G filters rotated from -alpha to alpha.
        self.M=M
        self.N=N 
        self.G=G
        self.a=alpha
        self.k = filtersize
        
        #self.conv1 = nn.Conv2d(self.M, self.M*self.N*self.G, filtersize, groups=self.M, padding=padding) # in, out, kernel size, groups as in paper
        
        # Lets follow the vocabluary of the paper and call:
        # w: the unrotated filters / "templates" (the ones we want to learn)
        # gw: the rotated filters / "transformed templates" (the G rotated versions of those filters)
        # There will be a total of:
        #   M*N templates
        #   M*N*G transformed templates
 
        # Create the "templates" as torch.nn.Parameter as described here:
        # http://pytorch.org/tutorials/advanced/numpy_extensions_tutorial.html
        # http://pytorch.org/docs/0.3.1/nn.html#parameters
        # Rondomly initialize weights: # TODO: Maybe there is a better way to initialize
        w = Parameter(torch.randn(M*N, k, k))  # size(w) = ( M*N x k x k )
        
        # From every "template" derive G "transformed templates" by applying rotation:
        # Variable(torch.randn(8,4,3,3))
        # Step 1 – Create the G transformation matrices we whish to rotate each "template" by:
        # TODO: Copy from Cat
        g = get_rotated_kernels(w, self.G, -alpha, alpha) #Variable(torch.randn(G,2,3)) # size(g) = ( G x 2 x 3 ) # TODO
        
        # Step2 – Apply the transformations:
        # Step 2.1 – Create the flow fields, describing the transformations.
        s = torch.Size((G, M*N, k, k)) # Desired output size
        flow_field  = torch.nn.functional.affine_grid(g, s) # size(flow_field) = (G, k, k, 2), one translation vector (or maybe coordinate; we don't know nor care) per each of the G rotation matrices.
        
        # Step 2.2 – Apply the flow_fields. Each flow_field will be applied to each channle of the input / each "template".
        # Each flow_field is of (G x M*N, k, k)
        #a grid. For each rotation, there will be one flow field.
        # Repeat w along the first dimension:
        w_rep = w.unsqueeze(0) # Add an empty first dimension. size(w_rep) = ( 1 x M*N, k, k)
        w_rep = w_rep.expand(G, -1, -1, -1) # Repeat along the singular first dimension G times. size(w_rep) = ( G x M*N, k, k)
              
        # Convolution:
        # Use the functional (torch.nn.functional.conv2d) instead of Module
        # (torch.nn.conv2d), becuse we don't want trainable parameters exept the
        # template vector defined above:
        out_conv = F.conv2d(inputs, filters, padding=1)    
        
        
        # Those two layers are the same as in the vanilla NPTN by ??? et.al. 20??
        self.maxpool3d = nn.MaxPool3d((self.G, 1, 1)) 
        self.meanpool3d = nn.AvgPool3d((self.M, 1, 1)) # Is that the right pooling? - AvgPool3d?
        
        self.permutation = make_permutation(self.M, self.N)

    def forward(self, x):
        #print('\nShape of x ', x.size())
                
        # The Tutorial on "Spatial transformer networks" has code showing the
        # usage of affine_grid and grid_sample modules.
        # http://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html#depicting-spatial-transformer-networks
        # http://pytorch.org/docs/0.3.1/nn.html?highlight=affine%20grid#torch.nn.functional.affine_grid
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
    
net = NewNPTN(M, N, G, alpha = alpha, filtersize = filtersize, padding=0)


################################################
M = 3
N = 2
G = 5
a = 5
k = 3
w = nn.Parameter(torch.randn(M*N, k, k).type(dtype) )  
################################################

'''
# How view can be used in the nn.Sequential pipeline:
# https://discuss.pytorch.org/t/equivalent-of-np-reshape-in-pytorch/144/6


#https://discuss.pytorch.org/t/is-indexing-in-pytorch-integrated-in-autograd/6992
Does the [] indexing get tracked by autograd?
 Yes
# Maybe this is realvant to: http://pytorch-zh.readthedocs.io/en/latest/autograd.html#torch.autograd.Function.mark_shared_storage
'''