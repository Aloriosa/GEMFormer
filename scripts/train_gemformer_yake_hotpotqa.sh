#!/bin/bash

cd ./gemformer

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"; accelerate launch --num_processes=8 --multi_gpu --main_process_port=20525 train_gemformer_yake_mem.py