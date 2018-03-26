#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 10:50:51 2018

Results for NPTN Experiments
"""

import numpy as np
import matplotlib.pyplot as plt


print(' ### Experiments with two-layered batch-normalized CNNs vs NPTNs ### ')
print('kernelsize = 5m after 300 epochs of training')
print('NLLL loss')


NPTN_paras = [(48,1), (24,2), (16,3)]
NPTN_losses = [0.93679728,0.9056078843712807,0.9312066503047943]
NPTN_accs = [67.95,68.97,67.55]

x = [1,2,3]
plt.plot(x, NPTN_losses, marker='p', label='NPTN')
plt.plot([1],[0.92783253], color='r', marker='p', label='CNN')  #  TODO insert actual value
plt.xlabel('(N,G)')
plt.ylabel('NLLL loss')
plt.xticks(x, ['(48,1)','(24,2)','(16,3)'])
plt.legend()
plt.show()

print('Accuracy (Not reported in original paper)')
plt.plot([1,2,3], NPTN_accs, marker='p', label='NPTN')
plt.plot([1],[68.29], color='r', marker='p', label='CNN')
plt.xlabel('(N,G)')
plt.xticks(x, ['(48,1)','(24,2)','(16,3)'])
plt.ylabel('Accuracy')
plt.legend()
plt.show()

print('Possible issues: difference train test loss?, no train acc reported')

print('\n\n ############################################################# \n\n')

print('Rotation Experiments on MNIST')
print('Kernelsize=5, Rotation=60?')
print('after 300 epochs')
print('NPTN(18,2)  Accuracy 96.32%, Loss = 0.12038491313457489')
print('CNN(36)     Accuracy 96.28%, Loss = 0.11692798318862915' )
print('\nKernelsize=5, Rotation=90')
print('CNN(36)    Accuracy 94%, Loss = 0.1816810576915741')
print('NPTN(18,2) Accuracy 94%, Loss = 0.2116')

print('\n\n')
print('NO HORIZONTAL FLIPPING Rotation Experiments on MNIST, BS32')

print('Rotation=90')
print('NPTN(18,2): Test NLLLoss =  0.0738690174460411,  Acc 97.72 %')
print('CNN(36):    Test NLLLoss =  0.07362796711921692, Acc 97.69 %')
print('\nRotation=60')
print('CNN(36) Test Nllloss = 0.0559881, Acc = 98.43')
print('NPNT(12,3) Test Nllloss = 0.04085, Acc = 98.75')

print('\n\nISSUES: Higher test acc/ lower test loss \n')

print('Experiment proposals:')
print('18,2 NPTN rot 60)
print('less important')
print('-- 12,3 NPTN rot 90')


print('\n\n ############################################################# \n\n')
      
print('Three layered experiments')



print('\n\n ############################################################# \n\n')
      
print('Rot Net experiments')

print('RotNet(12,3) MNIST 60 rot,alpha=60 Test NLLloss = 0.0888415, Acc=97.18 ')

print('\n\nScheduled:')
print('CIFAR comparism')

print('\n\nExperiment proposals')
print('- rot 90 comparism' )
print('')

print('\n\n ############################################################# \n\n')

print('Experiments where training set is not rotated')


print('\n\nScheduled:')
print('NO train rotation rot 60')


print('\n\nExperiment proposals')
print('Baseline CNN no train rotation rot 60')
print('rotnet higher G for rot 60')
print('less important')
print('-- rotnet 12,3 lower alpha')






# is MNIST 28*28?!
# what you mean no rotation for test set?
# baseline experiment?






