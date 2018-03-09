#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 10:50:51 2018

Results for NPTN Experiments
"""

import numpy as np
import matplotlib.pyplot as plt

print('Experiments with two-layered batch-normalized CNNs vs NPTNs')
print('kernelsize = 5m after 300 epochs of training')
print('NLLL loss')


NPTN_paras = [(48,1), (24,2), (16,3)]
NPTN_losses = [0.9322558387160301,0.9056078843712807,0.9312066503047943]
NPTN_accs = [67,68,67]

x = [1,2,3]
plt.plot(x, NPTN_losses, marker='p', label='NPTN')
plt.plot([1],[1.00062], color='r', marker='p', label='CNN')  #  TODO insert actual value
plt.xlabel('(N,G)')
plt.ylabel('NLLL loss')
plt.xticks(x, ['(48,1)','(24,2)','(16,3)'])
plt.legend()
plt.show()

print('Accuracy (Not reported in original paper)')
plt.plot([1,2,3], NPTN_accs, marker='p', label='NPTN')
plt.plot([1],[65], color='r', marker='p', label='CNN')
plt.xlabel('(N,G)')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



print('Rotation Experiments on MNIST')
print('Kernelsize=5, Rotation=60?')
print('NPTN(18,2)  Accuracy 96.32%, Loss = 0.12038491313457489')
print('CNN(36) Accuracy 96.26 %, Loss = 0.11692798318862915' )



# is MNIST 28*28?!
# what you mean no rotation for test set?
# baseline experiment?






