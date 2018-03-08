#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 09:33:35 2018

"""
print('Training the Baseline 2-layered CNN on CIFAR10')


import numpy as np
import matplotlib.pyplot as plt
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
parser.add_argument('-conv_1', '--conv_1_features', default = 48, type=int)
parser.add_argument('-conv_2', '--conv_2_features', default = 16, type=int)
parser.add_argument('-k', '--kernel_size', default = 5, type=int)
parser.add_argument('-b', '--batch_size', default = 4, type=int)
parser.add_argument('-o', '--out_dir', default = "output", type=str)

args = parser.parse_args()
conv_1_features = args.conv_1_features
conv_2_features = args.conv_2_features
kernel_size = args.kernel_size
out_dir = args.out_dir
batch_size = args.batch_size

ts = time.time()
timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
spec_string = str(timestamp) + '_CNN_BASELINE_' + ' Conv' + str(conv_1_features) + " Conv2" + str(conv_2_features) + " Kernel" + str(kernel_size)
out_dir = args.out_dir
experiment_out_dir = os.path.join(out_dir, spec_string)

if not os.path.exists(out_dir):
   os.makedirs(out_dir)
if not os.path.exists(experiment_out_dir):
   os.makedirs(experiment_out_dir)

csv_file_name = os.path.join( experiment_out_dir, spec_string + ".csv")
txt_file_name = os.path.join( experiment_out_dir, spec_string + ".txt")
txt_file = open(txt_file_name, "w", 1)


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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#######    Opening Connection to Visdom server and initialize plots   #########
viz = Visdom()


winCNN = viz.line(
    Y=np.array([0]), name='training',
    opts=dict(
            fillarea=False,
            showlegend=True,
            width=800,
            height=800,
            xlabel='Epochs',
            ylabel='Loss',
            #ytype='log',
            title=spec_string + ' NLLLoss',
            marginleft=30,
            marginright=30,
            marginbottom=80,
            margintop=30,
        )
)
    
winCNNACC = viz.line(
    Y=np.array([0]), name='test',
    opts=dict(
            fillarea=False,
            showlegend=True,
            width=800,
            height=800,
            xlabel='Epochs',
            ylabel='Acurracy',
            #ytype='log',
            title=spec_string + ' Accuracy',
            marginleft=30,
            marginright=30,
            marginbottom=80,
            margintop=30,
        )
)

#############        Network definition       ####################




if use_cuda:
    net.cuda()

############## Chooses optimizer and loss  ##############

criterion = nn.NLLLoss()  #TODO which things here?!
optimizer = optim.SGD(net.parameters(), lr=0.1)


############## Train the network  ######################


stat_epoch = list()
stat_batch = list()
stat_loss = list()


num_epochs = 300 # paper: 300


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
                win=winCNN,
                name='training',
                update='append',
                opts=dict(showlegend=True)
            )
            
            running_loss = 0.0



def validation(epoch):
    # measure accuracy (not in paper though, so could be removed), currently not working
    net.eval() # sets network into evaluation mode, might make difference for BatchNorm
    
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

        running_loss += loss.data[0] # * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)

        correct += (predicted == labels.data).sum()
    print('----------------------------------------------')
    print('----------------------------------------------', file=txt_file)
    print('Epoch ', epoch)
    print('Epoch ', epoch, file=txt_file)
    accuracy = (100 * correct / total)
    print('Accuracy of the NPTN network on the 10000 test images: %d %%' % accuracy)
    print('Accuracy of the NPTN network on the 10000 test images: %d %%' % accuracy, file=txt_file)
    print('NLLLoss = ', running_loss /(testloader.dataset.test_data.shape[0]/batch_size))
    print('NLLLoss = ', running_loss /(testloader.dataset.test_data.shape[0]/batch_size),
          file=txt_file)
    
    # update plot
    viz.line(
        X=np.array([epoch + 1]), 
        Y=np.array([running_loss /(testloader.dataset.test_data.shape[0]/batch_size)]),
        win=winCNN,
        name='test',
        update='append',
    opts=dict(showlegend=True))

    viz.line(
        X=np.array([epoch + 1]), 
        Y=np.array([100 * correct / total]),
        win=winCNNACC,
        name='test',
        update='append',
    opts=dict(showlegend=True))
    
    net.train()  # set network back in training mode
    return accuracy

################       Test the network        ##########################


# (taken from tutorial)
best_accuracy = 0.0
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
    if (epoch + 1) % 1 == 0:
        accuracy = validation(epoch)
        # Save the model:
        if ( accuracy > best_accuracy ):
            best_accuracy = accuracy
            model_file_name = os.path.join( experiment_out_dir, spec_string + ".CNN_model")
            torch.save(net, model_file_name)

print('Finished Training')

net.eval() # set to evaluation mode

# Save Data to CSV
stats_df = pd.DataFrame(
{'epoch': stat_epoch,
'batch': stat_batch,
'loss': stat_loss
})
stats_df.to_csv(csv_file_name)
txt_file.close()

# Save the model:
model_file_name = os.path.join( experiment_out_dir, spec_string + ".final_CNN_model")
torch.save(net, model_file_name)
