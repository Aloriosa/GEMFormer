#!/bin/bash

cd ./gemformer

export CUDA_VISIBLE_DEVICES="0"; accelerate launch --num_processes=1 --main_process_port=20526 val_gemformer_yake_mem.py \
--experiment_dir '../yake_mem_2wikimhqa' \
--val_dataset_path '../2wiki2mhqa_yake_mem_val_dataset' \
--raw_val_dataset_path '../2wiki2mhqa_val_examples_with_special_seps.pkl' \
--get_top_2_sp_para False
