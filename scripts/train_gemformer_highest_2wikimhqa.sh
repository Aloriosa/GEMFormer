#!/bin/bash

cd ./gemformer

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"; accelerate launch --num_processes=8 --multi_gpu --main_process_port=20525 train_gemformer.py \
--output_dir '../gemformer_highest_2wikimhqa' \
--entropy_threshold None \
--question_type_num_labels 3 \
--ques_type_loss_weight 1.
--ans_loss_weight 1 \
--para_loss_weight 1. \
--sent_loss_weight 1. \
--train_dataset_path '../2wikimhqa_preprocessed_train_examples_512_multitask_stride20_one_doc_batched_without_zero_answer_pos_without_CoT_triplets' \
--raw_train_dataset_path '../2wikimhqa_train_examples_with_special_seps.pkl' \
--global_batch_size 32 \
--lr 3e-5 \
--weight_decay 0.01 \
--num_epochs 5 \
--warmup_steps None \
--warmup_fraction 0.1 \
--supp_bce_loss None \
--ans_ce_loss False \
--tokens_to_add None \
--max_num_answers 64 \
--max_num_paragraphs 10 \
--max_num_sentences 210
