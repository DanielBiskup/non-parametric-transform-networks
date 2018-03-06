
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 09:20:49 2018

first experiment

"""


import numpy as np
#import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from network import NPTN
import pandas as pd
import sys
import os # for os.path.join()
import datetime
import time

import argparse
from visdom import Visdom

parser = argparse.ArgumentParser(description='GLRC stage 1')
parser.add_argument('-conv_1', '--conv_1_features', default = 24, type=int)
parser.add_argument('-g', '--group_size', default = 2, type=int)
parser.add_argument('-k', '--kernel_size', default = 5, type=int)
parser.add_argument('-b', '--batch_size', default = 4, type=int)
parser.add_argument('-o', '--out_dir', default = "output", type=str)

args = parser.parse_args()
conv_1_features = args.conv_1_features
G = args.group_size
kernel_size = args.kernel_size
out_dir = args.out_dir
batch_size = args.batch_size
###############   Test if you can use the GPU   ################

use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
    print('Using CUDA')

###########   loading and preprocessing the data    ############




# load dataset CIFAR10, normalize, crop and flip as in paper
transform = transforms.Compose(
    [transforms.RandomCrop(32, padding=4),
     transforms.RandomHorizontalFlip(), 
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#######    Opening Connection to Visdom server and initialize plots   #########


viz = Visdom()

startup_sec = 1
while not viz.check_connection() and startup_sec > 0:
    time.sleep(0.1)
    startup_sec -= 0.1
assert viz.check_connection(), 'No connection could be formed quickly'


win = viz.line(
    Y=np.array([0]), name='training'
)


#############        Network definition       ####################

class twoLayeredNPTN(nn.Module):
    def __init__(self, N, G, filtersize):
        super(twoLayeredNPTN, self).__init__()
        self.final_layer_dim = (7-np.int(filtersize/1.7))**2   # works for filtersizes 3,5,7
        # first layer 
        self.N = N
        self.nptn = NPTN(3, N, G, filtersize)
        self.batchnorm = nn.BatchNorm2d(N)   # is 2d the right one?
        self.pool = nn.MaxPool2d(2)
        self.prelu = nn.PReLU()
        # second layer
        self.nptn2 = NPTN(N, N, G, filtersize)
        self.batchnorm2 = nn.BatchNorm2d(N) 
        self.prelu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(2)
         
        self.fc1 = nn.Linear(N * self.final_layer_dim, 10)

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
        
        x = x.view(-1, self.N * self.final_layer_dim)
        #print('shape second layer ', x.size())
        x = F.log_softmax(self.fc1(x), dim=1)
        #print('after softmax ', x.size())
        return x




netN24G2 = twoLayeredNPTN(conv_1_features, G, kernel_size)
net = netN24G2

if use_cuda:
    net.cuda()

ts = time.time()
timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
spec_string = str(timestamp) + ' Conv' + str(conv_1_features) + " Group" + str(G) + " Kernel" + str(kernel_size)
out_dir = args.out_dir
experiment_out_dir = os.path.join(out_dir, spec_string)

if not os.path.exists(out_dir):
   os.makedirs(out_dir)
if not os.path.exists(experiment_out_dir):
   os.makedirs(experiment_out_dir)

csv_file_name = os.path.join( experiment_out_dir, spec_string + ".csv")
txt_file_name = os.path.join( experiment_out_dir, spec_string + ".txt")
txt_file = open(txt_file_name, "w" )
############## Chooses optimizer and loss  ##############

criterion = nn.NLLLoss()   #TODO which things here?!
optimizer = optim.SGD(net.parameters(), lr=0.1)


############## Train the network  ######################


num_epochs = 300 # paper: 300

stat_epoch = list()
stat_batch = list()
stat_loss = list()

def training_epoch(epoch):
    running_loss = 0.0
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
        if i % 500 == 499:
            stat_epoch.append(epoch + 1)
            stat_batch.append(i + 1)
            stat_loss.append(running_loss / 500)
            print('[%d, %5d] loss: %.3f' %
                  (stat_epoch[-1], stat_batch[-1], stat_loss[-1]))
            print('[%d, %5d] loss: %.3f' %
                  (stat_epoch[-1], stat_batch[-1], stat_loss[-1]), file = txt_file)
            sys.stdout.flush()

            # update plot
            viz.line(
                X=np.array([epoch + i/(trainloader.dataset.train_data.shape[0]/batch_size)]),
                Y=np.array([stat_loss[-1]]),
                win=win,
                name='training',
                update='append',
                opts=dict(showlegend=True)
            )

            running_loss = 0.0


def validation(epoch):
    # measure accuracy (not in paper though, so could be removed), currently not working
    correct = 0
    total = 0
    running_loss = 0.0

    for data in testloader:
        images, labels = data
        if use_cuda:
            images, labels = Variable(images.cuda()), Variable(labels.cuda())
        else:
            images, labels = Variable(images), Variable(labels)

        outputs = net(images)
        loss = criterion(outputs, labels)

        running_loss += loss.data[0] * images.size(0)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)

        correct += (predicted == labels.data).sum()
    print('----------------------------------------------')
    print('----------------------------------------------', file=txt_file)
    print('Epoch ', epoch)
    print('Epoch ', epoch, file=txt_file)
    print('Accuracy of the NPTN network on the 10000 test images: %d %%' % (
        100 * correct / total))
    print('Accuracy of the NPTN network on the 10000 test images: %d %%' % (
        100 * correct / total), file=txt_file)
    print('Cross entropy loss = ', running_loss /testloader.dataset.test_data.shape[0])
    print('Cross entropy loss = ', running_loss /testloader.dataset.test_data.shape[0],
          file=txt_file)
    
    # update plot
    viz.line(
        X=np.array([epoch]), 
        Y=np.array([running_loss /testloader.dataset.test_data.shape[0]]),
        win=win,
        name='test',
        update='append',
    opts=dict(showlegend=True))

# (taken from tutorial)
for epoch in range(num_epochs):  # loop over the dataset multiple times

    if epoch == 150:
        optimizer = optim.SGD(net.parameters(), lr=0.01)
        print('Learning rate adapted') # TODO change learning rate (optimizer? after certain iterations)
    if epoch == 225:
        optimizer = optim.SGD(net.parameters(), lr=0.001)
        print('Learning rate adapted')
    
    # call training epoch once
    training_epoch(epoch)
    
    
    
    # call validation batch every 5th epoch
    if epoch + 1 % 5 == 0:
        validation(epoch)

print('Finished Training')

# Save Data to CSV
stats_df = pd.DataFrame(
{'epoch': stat_epoch,
'batch': stat_batch,
'loss': stat_loss
})
stats_df.to_csv(csv_file_name)
txt_file.close()

