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