#!/bin/bash

cd ./gemformer

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"; accelerate launch --num_processes=8 --multi_gpu --main_process_port=20525 train_gemformer_yake_mem.py \
--output_dir '../yake_mem_2wikimhqa' \
--question_type_num_labels 3 \
--ques_type_loss_weight 1. \
--ans_loss_weight 1. \
--para_loss_weight 1. \
--sent_loss_weight 1. \
--train_dataset_path '../2wiki2mhqa_yake_mem_train_dataset' \
--global_batch_size 32 \
--lr 3e-5 \
--weight_decay 0.01 \
--num_epochs 5 \
--warmup_steps None \
--warmup_fraction 0.1 \
--supp_bce_loss None \
--ans_ce_loss False \
--tokens_to_add None