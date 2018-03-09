#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 py3.5 experiments.py -c MNIST_rot_60.yaml &&
CUDA_VISIBLE_DEVICES=0 py3.5 experiments.py -c MNIST_rot_90.yaml 
