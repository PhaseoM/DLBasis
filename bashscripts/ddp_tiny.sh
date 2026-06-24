  #! /bin/bash

CUDA_VISIBLE_DEVICES=2,3,4,5 \
torchrun --standalone --nproc_per_node=4 \
main.py -c tiny.toml