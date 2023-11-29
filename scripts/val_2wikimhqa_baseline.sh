#!/bin/bash

cd ./gemformer

export CUDA_VISIBLE_DEVICES="0"; accelerate launch --num_processes=1 --main_process_port=20526 val_roberta_baseline.py \
--experiment_dir '../roberta_baseline_2wikimhqa' \
--val_dataset_path '../2wikimhqa_preprocessed_val_examples_512_multitask_stride20_one_doc_batched_without_zero_answer_pos_without_CoT_triplets' \
--raw_dataset_validation_path '../2wikimhqa_val_examples_with_special_seps.pkl' \
--get_top_2_sp_para False
