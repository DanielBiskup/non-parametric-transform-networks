#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 py3.5 experiments.py -c E05_CNN_48_CIFAR_rot_0.yaml &&
CUDA_VISIBLE_DEVICES=0 py3.5 experiments.py -c E09_NPTN_M12_G4_CIFAR_rot_0.yaml &&
CUDA_VISIBLE_DEVICES=0 py3.5 experiments.py -c E21_NPTN_N7_G5_MNIST_rot60.yaml &&
CUDA_VISIBLE_DEVICES=0 py3.5 experiments.py -c E21_NPTN_N7_G5_MNIST_rot90.yaml &&
CUDA_VISIBLE_DEVICES=0 py3.5 experiments.py -c E22_rotNet_N12_G3_MNIST_alpha60_rotTrain90_rotTest90.yaml &&
CUDA_VISIBLE_DEVICES=0 py3.5 experiments.py -c E23_rotNet_N12_G3_MNIST_alpha90_rotTrain60_rotTest60.yaml &&
CUDA_VISIBLE_DEVICES=0 py3.5 experiments.py -c E23_rotNet_N12_G3_MNIST_alpha90_rotTrain90_rotTest90.yaml
