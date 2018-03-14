#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 py3.5 experiments.py -c 3_layer_nptn_32_48_24_2_k5.yaml &&
CUDA_VISIBLE_DEVICES=0 py3.5 experiments.py -c 3_layer_nptn_32_32_24_3_k5.yaml &&
CUDA_VISIBLE_DEVICES=0 py3.5 experiments.py -c 3_layer_nptn_24_32_16_4_k5.yaml &&
CUDA_VISIBLE_DEVICES=0 py3.5 experiments.py -c 3_layer_nptn_48_89_16_1_k5.yaml &&

