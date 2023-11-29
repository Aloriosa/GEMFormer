#!/bin/bash

cd ./gemformer

export CUDA_VISIBLE_DEVICES="0"; accelerate launch --num_processes=1 --main_process_port=20526 val_roberta_baseline.py \
--experiment_dir '../roberta_baseline_musique' \
--val_dataset_path '../musique_preprocessed_val_examples_512_allenai_style_multitask_stride20_one_doc_batched_without_zero_answer_pos' \
--raw_dataset_validation_path '../musique_val_examples_allenai_style_with_para_seps.pkl'
--get_top_2_sp_para None