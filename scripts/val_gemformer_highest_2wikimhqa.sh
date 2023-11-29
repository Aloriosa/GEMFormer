#!/bin/bash

cd ./gemformer

export CUDA_VISIBLE_DEVICES="0"; accelerate launch --num_processes=1 --main_process_port=20526 val_gemformer.py \
--experiment_dir '../gemformer_highest_2wikimhqa' \
--val_dataset_path '../2wiki2mhqa_preprocessed_val_examples_512_multitask_stride20_one_doc_batched_without_zero_answer_pos_without_CoT_triplets' \
--raw_val_dataset_path '../2wiki2mhqa_val_examples_with_special_seps.pkl' \
--get_top_2_sp_para False 
