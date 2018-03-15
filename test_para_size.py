#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 14:26:08 2018

find filter values
"""

def cnn_paras(N1,N2,N3=16):
    return 3*N1 + N1*N2 + N2*N3

def nptn_paras(N1,N2,N3,G):
    return 3*G*N1 + N1*G*N2 + N2*G*N3


def cnn_48(N2, N3=16):
    return 3*48 + 48*N2 + N2*N3

def cnn_89(N2, N3=16):
    return 3*89 + 89*N2 + N2*N3

def NPTN_48_3(N2=48, N3=16):
    return 3 * 48 * 3 + 3*48*N2 + 3*N2*N3


def cnn_104(N2, N3=16):
    return 3*104 + 104*N2 + N2*N3

def NPTN_48_4(n2, n3=16):
    return 3*48*4 + 48*4*n2 + n2*n3

def cnn_118(n2=89, n3=16):
    return 3*118 + 118*n2 + n2*n3

def NPTN_48_5(n2=48, n3=16):
    return 3*48*5 + 48*5*n2 + n2*n3


print(cnn_paras(48,89,16)) #5840

print(nptn_paras(48,89,16,1))
print(nptn_paras(48,40,16,2)) # 5408
print(nptn_paras(32,48,24,2)) #5568
print(nptn_paras(32,32,24,3)) #5664
print(nptn_paras(24,32,16,4)) # 5408
print(nptn_paras(32,24,24,4)) #5760
print(nptn_paras(16,32,16,4))
