#!/bin/bash

cd ./gemformer
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"; accelerate launch --num_processes=8 --multi_gpu --main_process_port=20525 train_gemformer.py \
--output_dir '../gemformer_highest_musique' \
--entropy_threshold None \
--question_type_num_labels None \
--ques_type_loss_weight None
--ans_loss_weight 1 \
--para_loss_weight 1. \
--sent_loss_weight None \
--train_dataset_path '../musique_preprocessed_train_examples_512_allanai_style_multitask_stride20_one_doc_batched_without_zero_answer_pos' \
--raw_train_dataset_path '../musique_train_examples_allenai_style_with_para_seps.pkl' \
--global_batch_size 12 \
--lr 2e-5 \
--weight_decay 0. \
--num_epochs 3 \
--warmup_steps 1000 \
--warmup_fraction None \
--supp_bce_loss True \
--ans_ce_loss True \
--tokens_to_add ['[para]'] \
--max_num_answers 1 \
--max_num_paragraphs 20 \
--max_num_sentences 152