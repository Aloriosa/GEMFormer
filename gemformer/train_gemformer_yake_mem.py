import argparse
import os
import math
import json
import torch
from tqdm import tqdm
from datetime import datetime
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaTokenizerFast, default_data_collator
from datasets import load_from_disk
from accelerate import Accelerator
from torch.utils.data import DataLoader
from .modeling_gemformer import RobertaGEMFormer
from .utils import add_qa_evidence_tokens, ROBERTA_BASE_SPECIAL_TOKENS, NOT_CONTENT_WORD_VOCAB


parser = argparse.ArgumentParser(description="train")
parser.add_argument("--output_dir", type=str, default='../yake_mem_hotpotqa')
parser.add_argument("--tokenizer_name", type=str, default='roberta-base')
parser.add_argument("--question_type_num_labels", type=int, default=3)
parser.add_argument("--ques_type_loss_weight", type=float, default=10.)
parser.add_argument("--ans_loss_weight", type=float, default=1.)
parser.add_argument("--para_loss_weight", type=float, default=1.)
parser.add_argument("--sent_loss_weight", type=float, default=1.)
parser.add_argument("--train_dataset_path", type=str, default='../hotpotqa_yake_mem_train_dataset')
parser.add_argument("--global_batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--num_epochs", type=int, default=5)
parser.add_argument("--warmup_steps", type=int, default=1000)
parser.add_argument("--warmup_fraction", type=bool, default=False)
parser.add_argument("--supp_bce_loss", type=bool, default=None)
parser.add_argument("--ans_ce_loss", type=bool, default=False)
parser.add_argument("--tokens_to_add", nargs="*", type=str, default=None, help='if not empty, first element has to be paragraph marker token')


def main(args):

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    start_epoch = 0
    if any([i.startswith('ckpt_') for i in os.listdir(args.output_dir)]):
        last_ckpt_num = max([int(name.split('_')[-1]) for name in os.listdir(args.output_dir) if name.startswith('ckpt_') and os.path.isdir(args.output_dir + '/' + name)])
        if os.path.exists(args.output_dir + f'/ckpt_{last_ckpt_num}/pytorch_model.bin'):
            model_name = args.output_dir+ f'/ckpt_{last_ckpt_num}'
            start_epoch = last_ckpt_num + 1
            print('Loading existing checkpoint')
    else:
        model_name = args.tokenizer_name

    print(f"\n\nmodel_name = {model_name}\n\n")

    tokenizer = RobertaTokenizerFast.from_pretrained(args.tokenizer_name)
    if args.tokens_to_add is not None:
        tokenizer = add_qa_evidence_tokens(tokenizer, tokens_to_add=args.tokens_to_add)
        PARA_TOKEN = tokenizer.convert_tokens_to_ids(args.tokens_to_add[0])
        SENT_MARKER_END_TOKEN = None
    else:
        tokenizer = add_qa_evidence_tokens(tokenizer)
        PARA_TOKEN = tokenizer.convert_tokens_to_ids('<t>')
        SENT_MARKER_END_TOKEN = tokenizer.convert_tokens_to_ids('[/sent]')

    model = RobertaGEMFormer.from_pretrained(model_name,
                                             paragraph_marker_token=PARA_TOKEN,
                                             sentence_marker_token=SENT_MARKER_END_TOKEN,
                                             question_type_num_labels=args.question_type_num_labels,
                                             ques_type_loss_weight=args.ques_type_loss_weight,
                                             ans_loss_weight=args.ans_loss_weight,
                                             para_loss_weight=args.para_loss_weight,
                                             sent_loss_weight=args.sent_loss_weight,
                                             supp_bce_loss=args.supp_bce_loss,
                                             ans_ce_loss=args.ans_ce_loss
                                             )
    model.resize_token_embeddings(len(tokenizer))

    train_dataset = load_from_disk(args.train_dataset_path)
    train_dataset = train_dataset.remove_columns(['id'])

    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    #one sample per gpu due to span prediction loss
    batch_size = 1
    gradient_accumulation_steps = args.global_batch_size // (num_gpus * batch_size)
    print(f"\n\nbatch_size = {batch_size}, gradient_accumulation_steps = {gradient_accumulation_steps}\n\n")

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=batch_size,
        
    )

    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)

    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_epochs = args.num_epochs
    num_update_steps_per_epoch = len(train_loader) // gradient_accumulation_steps 
    num_training_steps = num_epochs * num_update_steps_per_epoch

    if args.warmup_fraction:
        warmup_steps = int(args.warmup_fraction * num_training_steps)
    else:
        warmup_steps = args.warmup_steps

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optim,
        num_warmup_steps=warmup_steps * gradient_accumulation_steps,
        num_training_steps=num_training_steps * gradient_accumulation_steps
    )

    device = accelerator.device
    model.to(device)

    train_loader, model, optim, lr_scheduler = accelerator.prepare(
        train_loader, model, optim, lr_scheduler)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_loader) / gradient_accumulation_steps)
    num_training_steps = num_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    num_epochs = math.ceil(num_training_steps / num_update_steps_per_epoch)
    print(f"after accelerate prepare: total_num_batches = {len(train_loader)}, num_training_steps = {num_training_steps}, num_epochs = {num_epochs}\n\n")
    total_num_batches = len(train_loader) // gradient_accumulation_steps
    total_batch_size = batch_size * accelerator.num_processes * gradient_accumulation_steps
    print(f"total_batch_size = {total_batch_size}, accelerator.num_processes = {accelerator.num_processes}")
    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)

    if accelerator.is_local_main_process:
        with open(args.output_dir+'/train_config.json', 'w+') as f:
            json.dump(
                {
                'tokenizer_name': args.tokenizer_name,
                'question_type_num_labels': args.question_type_num_labels,
                'ques_type_loss_weight': args.ques_type_loss_weight,
                'ans_loss_weight': args.ans_loss_weight,
                'para_loss_weight': args.para_loss_weight,
                'sent_loss_weight': args.sent_loss_weight,
                'supp_bce_loss': args.supp_bce_loss,
                'start_epoch': start_epoch,
                'tokenizer_len': len(tokenizer),
                'train_dataset_path': args.train_dataset_path,
                'global_batch_size': args.global_batch_size,
                'num_gpus': num_gpus,
                'batch_size': batch_size,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'lr': args.lr,
                'weight_decay': args.weight_decay,
                'num_epochs': num_epochs,
                'warmup_steps': warmup_steps,
                'num_training_steps': num_training_steps,
                'para_token': PARA_TOKEN,
                'sent_token': SENT_MARKER_END_TOKEN
            }, f, indent=2)

    num_batches = 0
    for epoch in range(start_epoch, num_epochs):
        model.train()

        for batch in train_loader:
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch['input_ids'][0],
                    attention_mask=batch['attention_mask'][0],
                    context_start_id=batch['context_start_id'][0],
                    start_positions=batch['start_positions'][0],
                    end_positions=batch['end_positions'][0],
                    question_type_labels=batch['question_type'][0][:1] if args.question_type_loss_weight is not None else None,
                    paragraph_labels=batch['supp_title_labels'][0] if 'supp_title_labels' in batch else batch['supp_para_labels'][0],
                    sentence_labels=batch['supp_sent_labels'][0] if args.sent_loss_weight is not None else None,
                    mem_tokens=batch['mem_tokens'][0]
                )
                loss = outputs.loss
                accelerator.backward(loss)
                optim.step()
                lr_scheduler.step()
                optim.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                num_batches += 1

        print(f"Epoch {epoch} ended, saving the ckpt")
        
        # Save and upload
        # create subfolder for each epoch ckpt
        curr_ckpt_path = args.output_dir + f'/ckpt_{epoch}'
        if not os.path.exists(curr_ckpt_path):
            os.makedirs(curr_ckpt_path, exist_ok=True)

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(curr_ckpt_path, 
                                        is_main_process=accelerator.is_main_process, 
                                        save_function=accelerator.save)

        if accelerator.is_main_process:
            tokenizer.save_pretrained(curr_ckpt_path)
        accelerator.save_state(curr_ckpt_path)
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
