
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 09:20:49 2018

first experiment

"""

import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import sys
import os
import datetime
import time
import argparse
from visdom import Visdom
import yaml
#from shutil import copyfile

## import the networks:
from network import twoLayeredNPTN
from network import threeLayeredNPTN
from network import twoLayeredCNN
from network import threeLayeredCNN

'''
parser = argparse.ArgumentParser(description='Experiment')
parser.add_argument('-n', '--network_type', default = none, type=str, help='choose \'nptn\' or \'cnn\'')
parser.add_argument('-e', '--experiment', default = 'nptn', type=str, help='))
parser.add_argument('-conv_1', '--conv_1_features', default = 24, type=int)
parser.add_argument('-conv_2', '--conv_2_features', default = 16, type=int)
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
net_type = args.network_type
'''

parser = argparse.ArgumentParser(description='Experiment')
parser.add_argument('-c', '--config', default = "3_layer_nptn_48_3_k5.yaml", type=str, help='path to a .yaml configuration file')
# parser.add_argument('-c', '--config', default = "x.yaml", type=str, help='path to a .yaml configuration file')
parser.add_argument('-o', '--out_dir', default = "output", type=str)
parser.add_argument('-n', '--name', default = "yaml", type=str, help='yaml: Will use the yaml file name for folder and file names. n will use number of layers as file name')
args = parser.parse_args()

ts = time.time()
timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
timestamp = timestamp.replace(':','_').replace(' ','_').replace('-','_')

yaml_file_name = args.config
with open(yaml_file_name, 'r') as stream:
    try:
        d = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

def is_set(d,key):
    if key in d and d[key] != 0:
        return True
    else:
        return False

# Default values:
if 'batchsize' in d:
    batch_size = d['batchsize']
else:
    batch_size = 4

# Which data set to use?
ss = str(timestamp)
if d['dataset'] == 'mnist':
    M = 1
    ss = ss + '_mnist_1M'
    image_size = 28
elif d['dataset'] == 'cifar10':
    M = 3 
    ss = ss + '_cifar10_3M'
    image_size = 32

if d['type'] == 'nptn':
    ss = ss + '_nptn_' + str(d['layers']) + 'layers'
    if   d['layers'] == 2:
        net = twoLayeredNPTN(d['n1'], d['g'], d['filtersize'], in_channels=M)
        ss = ss + str(d['n1']) + 'N1_' + str(d['g']) + 'G_' + str(d['filtersize']) + "Kernel"
    elif d['layers'] == 3:
        net = threeLayeredNPTN(filtersize=d['filtersize'], G=d['g'] ,n1=d['n1'], n2=d['n2'], n3=d['n3'], input_channel=M)
        ss = ss + str(d['n1']) + 'N1_' + str(d['n2']) + 'N2_'+ str(d['n3']) + 'N3_'+ str(d['g']) + 'G_' + str(d['filtersize']) + "Kernel"

elif d['type'] == 'cnn':
    ss = ss + '_cnn_' + str(d['layers']) + 'layers'
    if d['layers'] == 2:
        net = twoLayeredCNN(d['filtersize'], in_channels=M, N1=d['n1'], N2=d['n2'])
        ss = ss + str(d['n1']) + 'N1_' + str(d['n2']) + 'N2_'+ str(d['filtersize']) + "Kernel"
    elif d['layers'] == 3:
        net = threeLayeredCNN(filtersize=d['filtersize'], n1=d['n1'], n2=d['n2'], n3=d['n3'], input_channel=M)
        ss = ss + str(d['n1']) + 'N1_' + str(d['n2']) + 'N2_'+ str(d['n3']) + 'N3_'+ str(d['filtersize']) + "Kernel"
        
if is_set(d, 'rotation_train'):
    ss = ss + '__rotation' + str(d['rotation_train'])

if args.name == 'yaml':
    ss = str(timestamp) + '_FRIDAY_' + args.config.replace('.','_')
    
spec_string = ss

###########   loading and preprocessing the data    ############

# TODO add num_workers

# load dataset CIFAR10, normalize, crop and flip as in paper

### Training Data Transforms
transform_train_list = [
     transforms.RandomHorizontalFlip()]
    
# Train:Translation
if is_set(d,'translation_train'):
    translation_train = d['translation_train']
    transform_train_list.append( transforms.RandomCrop(image_size, padding=translation_train) ) 
else:
    pass

# Train:Rotation
if is_set(d,'rotation_train'):
    rotation_train = d['rotation_train']
    transform_train_list.append( transforms.RandomRotation(rotation_train) ) 
else:
    pass

transform_train_list.append(transforms.ToTensor())

# Train:Normalization
if d['dataset'] == 'cifar10':
    transform_train_list.append( transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
elif d['dataset'] == 'mnist':
    transform_train_list.append(transforms.Normalize((0.1307,), (0.3081,)))

transform_train = transforms.Compose( transform_train_list )

### Test Data Transforms
transform_test_list = [
     transforms.RandomHorizontalFlip()]

# Don't use translation or rotation during test
if is_set(d,'rotation_test'):
    rotation_test = d['rotation_test']
    transform_test_list.append( transforms.RandomRotation(rotation_test) )
else:
    pass

# Test:Translation       
if is_set(d,'translation_test'):
    translation_test = d['translation_test']
    transform_test_list.append( transforms.RandomCrop((image_size,image_size), padding=translation_test) ) 
else:
    pass

transform_test_list.append(transforms.ToTensor()) 

# Test:Normalization
if d['dataset'] == 'cifar10':
    transform_test_list.append( transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
elif d['dataset'] == 'mnist':
    transform_test_list.append(transforms.Normalize((0.1307,), (0.3081,)))

transform_test = transforms.Compose( transform_test_list )

##### Load Data for Train and Test:
num_workers = 4

if d['dataset'] == 'cifar10':
    print('Using CIFAR10')
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)
elif d['dataset'] == 'mnist':
    print('Using MNIST')
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                                download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

###Shared code: ###############################################################
###############################################################################

## Set up files and directories for output:
out_dir = args.out_dir
experiment_out_dir = os.path.join(out_dir, spec_string)

#copyfile(yaml_file_name, experiment_out_dir + '/')

if not os.path.exists(out_dir):
   os.makedirs(out_dir)
if not os.path.exists(experiment_out_dir):
   os.makedirs(experiment_out_dir)


txt_file_name = os.path.join( experiment_out_dir, spec_string + ".txt")
csv_file_name = os.path.join( experiment_out_dir, spec_string + ".csv")
validation_csv_file_name = os.path.join( experiment_out_dir, spec_string + "_VALIDATION.csv")

csv_file = open(csv_file_name, "w", 1)
txt_file = open(txt_file_name, "w", 1)

# Save YAML dictionary to file:            
print(str(d), file = txt_file)

validation_csv_file = open(validation_csv_file_name, "w", 1)

print('batch,epoch,loss', file=csv_file)
print('epoch,accuracy,validationNLLLoss', file=validation_csv_file)

###############   Test if you can use the GPU   ################

use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
    print('Using CUDA')

#######    Opening Connection to Visdom server and initialize plots   #########
viz = Visdom(env = spec_string)

'''
startup_sec = 1
while not viz.check_connection() and startup_sec > 0:
    time.sleep(1)
    startup_sec -= 0.1
assert viz.check_connection(), 'No connection could be formed quickly'
'''

winLoss = viz.line(
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
    
winACC = viz.line(
    Y=np.array([0]), name='test',
    opts=dict(
            fillarea=False,
            showlegend=True,
            width=800,
            height=800,
            xlabel='Epochs',
            ylabel='Accuracy',
            #ytype='log',
            title=spec_string + ' Accuracy',
            marginleft=30,
            marginright=30,
            marginbottom=80,
            margintop=30,
        )
)

if use_cuda:
    net.cuda()

############## Chooses optimizer and loss  ##############

criterion = nn.NLLLoss()   #TODO which things here?!
optimizer = optim.SGD(net.parameters(), lr=0.1)

############## Train the network  ######################

num_epochs = 300 # paper: 300

def training_epoch(epoch):
    running_loss = 0.0
    correct = 0
    
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
        
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels.data).sum()
        
        if i % 500 == 499:
            stat_epoch = epoch + 1
            stat_batch = i + 1
            stat_loss = running_loss / 500
            print('[%d, %5d] loss: %.3f' %
                  (stat_epoch, stat_batch, stat_loss))
            print('[%d, %5d] loss: %.3f' %
                  (stat_epoch, stat_batch, stat_loss), file = txt_file)
            sys.stdout.flush()
            print('%i,%i,%.3f' % (stat_epoch, stat_batch, stat_loss), file=csv_file)

            # update plot
            viz.line(
                X=np.array([epoch + i/(trainloader.dataset.train_data.shape[0]/batch_size)]),
                Y=np.array([stat_loss]),
                win=winLoss,
                name='training',
                update='append',
                opts=dict(showlegend=True)
            )
            
            running_loss = 0.0
    accuracy = (100 * correct / trainloader.dataset.train_data.shape[0])  
    print('Accuracy of the network on the train images: %d %%' % accuracy)
    print('Accuracy of the network on the train images: %d %%' % accuracy, file=txt_file)

    viz.line(
        X=np.array([epoch + 1]), 
        Y=np.array([accuracy]),
        win=winACC,
        name='train',
        update='append',
    opts=dict(showlegend=True))
    

# TODO 
'''
def validation(epoch, loader):
    # measure accuracy (not in paper though, so could be removed), currently not working
    net.eval() # sets network into evaluation mode, might make difference for BatchNorm
    
    correct = 0
    total = 0
    running_loss = 0.0

    
    for data in loader:
        images, labels = data
'''

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
    NLLLoss = running_loss /(testloader.dataset.test_data.shape[0]/batch_size)
    print('Accuracy of the network on the 10000 test images: %d %%' % accuracy)
    print('Accuracy of the network on the 10000 test images: %d %%' % accuracy, file=txt_file)
    print('NLLLoss = ', NLLLoss)
    print('NLLLoss = ', NLLLoss,
          file=txt_file)

    #Save CSV: #
    print('%i,%.3f,%.3f' % (epoch, accuracy, NLLLoss ), file=validation_csv_file)
    
    # update plot
    viz.line(
        X=np.array([epoch + 1]), 
        Y=np.array([running_loss /(testloader.dataset.test_data.shape[0]/batch_size)]),
        win=winLoss,
        name='test',
        update='append',
    opts=dict(showlegend=True))

    viz.line(
        X=np.array([epoch + 1]), 
        Y=np.array([100 * correct / total]),
        win=winACC,
        name='test',
        update='append',
    opts=dict(showlegend=True))
    
    net.train()  # set network back in training mode
    return accuracy
    
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
            model_file_name = os.path.join( experiment_out_dir, spec_string + ".model")
            torch.save(net, model_file_name)

print('Finished Training')

net.eval() # set to evaluation mode

txt_file.close()
csv_file.close()
validation_csv_file.close()

# Save the model:
model_file_name = os.path.join( experiment_out_dir, spec_string + ".final_model")
torch.save(net, model_file_name)
