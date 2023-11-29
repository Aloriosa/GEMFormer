#!/bin/bash

cd ./gemformer

export CUDA_VISIBLE_DEVICES="0"; accelerate launch --num_processes=1 --main_process_port=20526 val_gemformer.py