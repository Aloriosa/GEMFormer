#!/bin/bash

cd ./gemformer

export CUDA_VISIBLE_DEVICES="0"; accelerate launch --num_processes=1 --main_process_port=20526 val_gemformer_yake_mem.py \
--experiment_dir '../yake_mem_musique' \
--val_dataset_path '../musique_yake_mem_val_dataset' \
--raw_val_dataset_path '../musique_val_examples_allenai_style_with_para_seps.pkl' \
--get_top_2_sp_para None
