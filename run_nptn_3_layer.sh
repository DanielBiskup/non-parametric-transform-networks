#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 py3.5 experiments.py -c rotNet_16_3_CIFAR_rot_-60_60.yaml &&
CUDA_VISIBLE_DEVICES=0 py3.5 experiments.py -c rotNet_12_3_MNIST_rot_-60_60_rot_60.yaml &&
CUDA_VISIBLE_DEVICES=0 py3.5 experiments.py -c rotNet_12_3_MNIST_rot_-60_60_rot_60_only_training.yaml

# more but less important

