#!/bin/bash

cd ./gemformer

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"; accelerate launch --num_processes=8 --multi_gpu --main_process_port=20525 train_roberta_baseline.py \
--output_dir '../roberta_baseline_musique' \
--question_type_num_labels None \
--question_type_loss_weight None \
--ans_loss_weight 1. \
--para_loss_weight 1. \
--sent_loss_weight None \
--train_dataset_path '../musique_preprocessed_train_examples_512_multitask_stride20_one_doc_batched_without_zero_answer_pos' \
--global_batch_size 12 \
--lr 2e-5 \
--weight_decay 0. \
--num_epochs 3 \
--warmup_steps 1000 \
--supp_bce_loss True \
--ans_ce_loss True \
--tokens_to_add ['[para]']