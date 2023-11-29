#!/bin/bash

cd ./gemformer
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"; accelerate launch --num_processes=8 --multi_gpu --main_process_port=20525 train_roberta_baseline.py \
--output_dir '../roberta_baseline_2wikimhqa' \
--question_type_loss_weight 1. \
--ans_loss_weight 1. \
--para_loss_weight 1. \
--sent_loss_weight 1. \
--train_dataset_path '../2wikimhqa_preprocessed_train_examples_512_multitask_stride20_one_doc_batched_without_zero_answer_pos_without_CoT_triplets' \
--global_batch_size 32 \
--lr 3e-5 \
--num_epochs 5 \
--warmup_fraction 0.1
