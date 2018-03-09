#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 15:18:33 2018

@author: lab
"""

# PyTorch Notes

# Prep for linear layer:
import torch
t = torch.IntTensor(4, 3, 5, 5).zero_()
t.view(t.size(0), -1)
