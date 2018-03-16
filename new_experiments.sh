#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 py3.5 experiments.py -c rotNet_7_5_MNIST_rot_-60_60_rot_60_only_training.yaml &&
CUDA_VISIBLE_DEVICES=0 py3.5 experiments.py -c NPTN12_3_MNIST_rot_90.yaml &&
CUDA_VISIBLE_DEVICES=0 py3.5 experiments.py -c NPTN18_2_MNIST_rot_60.yaml

