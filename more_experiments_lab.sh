#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python3.6 experiments.py -c rotNet_12_3_MNIST_rot_-60_60_rot_30.yaml &&
CUDA_VISIBLE_DEVICES=0 python3.6 experiments.py -c rotNet_12_3_MNIST_rot_-90_90_rot_30.yaml &&
CUDA_VISIBLE_DEVICES=0 python3.6 experiments.py -c nptn_18_2_MNIST_rot_-90_90.yaml &&
CUDA_VISIBLE_DEVICES=0 python3.6 experiments.py -c rotNet_7_5_MNIST_rot_-60_60_rot_60.yaml &&
CUDA_VISIBLE_DEVICES=0 python3.6 experiments.py -c NPTN_M48_G1_CIFAR_rot_0.yaml &&
CUDA_VISIBLE_DEVICES=0 python3.6 experiments.py -c NPTN_M24_G2_CIFAR_rot_0.yaml &&
CUDA_VISIBLE_DEVICES=0 python3.6 experiments.py -c NPTN_M16_G3_CIFAR_rot_0.yaml
