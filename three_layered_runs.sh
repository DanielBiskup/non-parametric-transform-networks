#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 py3.5 experiments.py -c 3_layer_nptn_48_3_k5.yaml &&
CUDA_VISIBLE_DEVICES=0 py3.5 experiments.py -c 3_layer_cnn_89_k5.yaml &&
CUDA_VISIBLE_DEVICES=0 py3.5 experiments.py -c 3_layer_nptn_48_4_k5.yaml &&
CUDA_VISIBLE_DEVICES=0 py3.5 experiments.py -c 3_layer_cnn_104_k5.yaml &&
CUDA_VISIBLE_DEVICES=0 py3.5 experiments.py -c 3_layer_nptn_48_5_k5.yaml &&
CUDA_VISIBLE_DEVICES=0 py3.5 experiments.py -c 3_layer_cnn_118_k5.yaml
