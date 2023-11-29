#!/bin/bash

cd ./gemformer

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"; accelerate launch --num_processes=8 --multi_gpu --main_process_port=20525 train_gemformer_yake_mem.py \
--output_dir '../yake_mem_musique' \
--question_type_num_labels None \
--ques_type_loss_weight None \
--ans_loss_weight 1. \
--para_loss_weight 1. \
--sent_loss_weight None \
--train_dataset_path '../musique_yake_mem_train_dataset' \
--global_batch_size 12 \
--lr 2e-5 \
--weight_decay 0. \
--num_epochs 3 \
--warmup_steps 1000 \
--warmup_fraction None \
--supp_bce_loss True \
--ans_ce_loss True \
--tokens_to_add ['[para]']