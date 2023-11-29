import argparse
import os
import math
import json
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaTokenizerFast, default_data_collator
from datasets import load_from_disk 
from datasets.arrow_dataset import Dataset
from accelerate import Accelerator
from torch.utils.data import DataLoader
from modeling_gemformer import RobertaGEMFormer
from utils import add_qa_evidence_tokens, pad_and_drop_duplicates, ROBERTA_BASE_SPECIAL_TOKENS, NOT_CONTENT_WORD_VOCAB


def generate_full_train_memory(accelerator, model, val_loader, 
                               paragraph_marker_token, sentence_marker_token,
                               output_dir=None, entropy_threshold=None,
                               max_mem_len=200):
    if entropy_threshold is not None:
        accelerator.print(f"memory generation: entropy_threshold = {entropy_threshold}")
    model.eval()
    inputs = []
    seq_logits_list = []
    start_list = []
    end_list = []
    mem_train_list = []
    entropy_percentiles = []
        
    for batch in tqdm(val_loader):
        with torch.no_grad():
            mem_tokens = []
            mem_inp_batch = batch['input_ids'][0][:, batch['context_start_id'][0][0]:].contiguous().view(1, -1)
            numpy_batch = mem_inp_batch.cpu().numpy()
            spec_tokens_found = np.intersect1d(numpy_batch, np.array(ROBERTA_BASE_SPECIAL_TOKENS)).tolist()
            bool_mask = mem_inp_batch == paragraph_marker_token
            if sentence_marker_token is not None:
                bool_mask = torch.logical_or(bool_mask, mem_inp_batch == sentence_marker_token)
            for i in spec_tokens_found:
                bool_mask = torch.logical_or(bool_mask, mem_inp_batch == i)

            outputs = model(
                    input_ids=batch['input_ids'][0],
                    attention_mask=batch['attention_mask'][0],
                    context_start_id=batch['context_start_id'][0],
                    return_lm_logits=True
            )
            sequence_output = outputs.lm_logits
            probs = torch.softmax(sequence_output, dim=-1)
            ue_batch = torch.sum(-probs * torch.log(torch.clamp(probs, 1e-8, 1)),
                                 dim=-1) / torch.log(torch.tensor(probs.shape[-1]))

            if output_dir is not None:
                padded_mem_probas = torch.nn.functional.pad(ue_batch,
                                                               (0, 4096 - ue_batch.shape[1]),
                                                               "constant", -1).to(batch['input_ids'].device)
            ue_batch_context = ue_batch[:, batch['context_start_id'][0][0]:].contiguous().view(1, -1)
            # nonempty entropy_threshold means the Low entropy memory
            if entropy_threshold is not None:
                # calculate actual threshold value for current document that is equal to the percentile of the document entropy
                if entropy_threshold > 1:
                    entropy_threshold = np.percentile(ue_batch_context.cpu().numpy()[0], entropy_threshold)
                not_content_word_mask = NOT_CONTENT_WORD_VOCAB[mem_inp_batch].to(ue_batch_context.device)
                special_sep_mask = torch.logical_or(not_content_word_mask, bool_mask)
                filtered_entropy = ue_batch_context + (special_sep_mask.to(ue_batch_context.device) * 1e9).long()
                mem_tokens = torch.gather(mem_inp_batch, 1, 
                                          torch.where(filtered_entropy < entropy_threshold)[-1].view(1, -1))
            else:
                # entropy_threshold = None means we fill memory with the max_mem_len tokens with the highest entropy values 
                special_sep_mask = 1 - bool_mask.long()

                filtered_entropy = ue_batch_context * special_sep_mask.to(ue_batch_context.device).long()
                #drop tokens and entropies that match filters
                gathered_filtered_entropy = torch.gather(filtered_entropy, 1,
                                                         torch.where(filtered_entropy > 0)[-1].view(1, -1))
                gathered_mem_inp_batch = torch.gather(mem_inp_batch, 1,
                                                      torch.where(filtered_entropy > 0)[-1].view(1, -1))
                # if the sequence after filtering is shorter than max_mem_len - take it all
                topk_positions = torch.topk(gathered_filtered_entropy,
                                              min(max_mem_len,
                                                  gathered_filtered_entropy.shape[-1]),
                                              sorted=False).indices.to(mem_inp_batch.device,
                                              dtype=torch.int64)
                mem_tokens = torch.gather(gathered_mem_inp_batch, 1, topk_positions)

            mem_tokens = torch.nn.functional.pad(mem_tokens, (0, max_mem_len - mem_tokens.shape[1]),
                                                    "constant", -1).to(batch['input_ids'].device)

        g_mem_tokens = accelerator.gather(mem_tokens)
        mem_train_list += [list(filter((-1).__ne__,
                                       i.cpu().numpy().tolist()))[:max_mem_len] for i in g_mem_tokens] 

        # for dynamic threshold based on the entropy percentile, collect exact threshold values
        if entropy_threshold > 1:
            g_entropy_threshold = accelerator.gather(
                torch.tensor([[entropy_threshold]]).to(mem_tokens.device))
            entropy_percentiles += [i.cpu().numpy().tolist() for i in g_entropy_threshold]

        # collect some data for further analysis
        if output_dir is not None: 
            g_inputs = accelerator.gather(batch['input_ids'][0])
            g_mem_probas = accelerator.gather(padded_mem_probas)
            g_start_idx = accelerator.gather(batch['context_start_id'][0])
            g_end_idx = accelerator.gather(batch['context_end_id'][0])

            inputs += [i.cpu().numpy().tolist() for i in g_inputs]
            seq_logits_list += [i.cpu().numpy().tolist() for i in g_mem_probas]
            start_list += [i.cpu().numpy().tolist() for i in g_start_idx]
            end_list += [i.cpu().numpy().tolist() for i in g_end_idx]

    if output_dir is not None:
        torch.save(inputs, output_dir + '/inputs_list.pkl')
        torch.save(seq_logits_list, output_dir + '/seq_logits_list.pkl')
        torch.save(start_list, output_dir + '/context_start_id_list.pkl')
        torch.save(end_list, output_dir + '/context_end_id_list.pkl')
    return mem_train_list, entropy_percentiles


parser = argparse.ArgumentParser(description="train")
parser.add_argument("--output_dir", type=str, default='../gemformer_highest_hotpotqa')
parser.add_argument("--max_mem_len", type=int, default=200, help='maximal memory length in tokens')
parser.add_argument("--entropy_threshold", type=float, default=None, help='None for Highest entropy memory of size max_mem_len or value from 0 to 1 for Low entropy memory with constant threshold or value > 1 for dynamic Low entropy memory with percentile-based threshold')
parser.add_argument("--tokenizer_name", type=str, default='roberta-base')
parser.add_argument("--question_type_num_labels", type=int, default=3)
parser.add_argument("--ques_type_loss_weight", type=float, default=10.)
parser.add_argument("--ans_loss_weight", type=float, default=1.)
parser.add_argument("--para_loss_weight", type=float, default=1.)
parser.add_argument("--sent_loss_weight", type=float, default=1.)
parser.add_argument("--train_dataset_path", type=str, default='../hotpotqa_preprocessed_train_examples_512_multitask_stride20_one_doc_batched_without_zero_answer_pos')
parser.add_argument("--raw_train_dataset_path", type=str, default='../hotpotqa_train_examples_with_special_seps.pkl')
parser.add_argument("--global_batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--num_epochs", type=int, default=5)
parser.add_argument("--warmup_steps", type=int, default=1000)
parser.add_argument("--warmup_fraction", type=bool, default=False)
parser.add_argument("--supp_bce_loss", type=bool, default=None)
parser.add_argument("--ans_ce_loss", type=bool, default=False)
parser.add_argument("--tokens_to_add", nargs="*", type=str, default=None, help='if not empty, first element has to be paragraph marker token')
# dataset-specific values for maximal possible number of answers, paragraphs and sentences per one document among train and valid data
parser.add_argument("--max_num_answers", type=int, default=64)
parser.add_argument("--max_num_paragraphs", type=int, default=10)
parser.add_argument("--max_num_sentences", type=int, default=150)


def main(args):

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    start_epoch = 0
    last_ckpt_num = None
    if any([i.startswith('ckpt_') for i in os.listdir(args.output_dir)]):
        last_ckpt_num = max(
            [int(name.split('_')[-1]) for name in os.listdir(args.output_dir) if name.startswith('ckpt_') and \
             os.path.isdir(args.output_dir + '/' + name)])
        if os.path.exists(args.output_dir + f'/ckpt_{last_ckpt_num}/pytorch_model.bin'):
            model_name = args.output_dir + f'/ckpt_{last_ckpt_num}'
            start_epoch = last_ckpt_num + 1
            print('Loading existing checkpoint')
    else:
        model_name = args.tokenizer_name

    tokenizer = RobertaTokenizerFast.from_pretrained(args.tokenizer_name)
    if args.tokens_to_add is not None:
        tokenizer = add_qa_evidence_tokens(tokenizer, tokens_to_add=args.tokens_to_add)
        PARA_TOKEN = tokenizer.convert_tokens_to_ids(args.tokens_to_add[0])
        SENT_MARKER_END_TOKEN = None
    else:
        tokenizer = add_qa_evidence_tokens(tokenizer)
        PARA_TOKEN = tokenizer.convert_tokens_to_ids('<t>')
        SENT_MARKER_END_TOKEN = tokenizer.convert_tokens_to_ids('[/sent]')


    
    model = RobertaGEMFormer.from_pretrained(
        model_name, 
        paragraph_marker_token=PARA_TOKEN,
        sentence_marker_token=SENT_MARKER_END_token,
        question_type_num_labels=args.question_type_num_labels,
        ques_type_loss_weight=args.ques_type_loss_weight,
        ans_loss_weight=args.ans_loss_weight,
        para_loss_weight=args.para_loss_weight,
        sent_loss_weight=args.sent_loss_weight,
        supp_bce_loss=args.supp_bce_loss,
        ans_ce_loss=args.ans_ce_loss
    )
    model.resize_token_embeddings(len(tokenizer))

    MAX_SEQ_LEN = model.config.max_position_embeddings - 2

    # model for estimation of uncertainty of predictions
    model_up = RobertaGEMFormer.from_pretrained(
        model_name, 
        paragraph_marker_token=PARA_TOKEN,
        sentence_marker_token=SENT_MARKER_END_token,
        question_type_num_labels=args.question_type_num_labels,
        ques_type_loss_weight=args.ques_type_loss_weight,
        ans_loss_weight=args.ans_loss_weight,
        para_loss_weight=args.para_loss_weight,
        sent_loss_weight=args.sent_loss_weight,
        supp_bce_loss=args.supp_bce_loss,
        ans_ce_loss=args.ans_ce_loss
    )
    model_up.resize_token_embeddings(len(tokenizer))
    
    train_dataset = load_from_disk(args.train_dataset_path)
    train_dataset = train_dataset.remove_columns(['id'])


    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    # one sample per gpu due to span prediction loss
    batch_size = 1    
    gradient_accumulation_steps = args.global_batch_size // (num_gpus * batch_size)


    train_set = train_dataset.remove_columns(['start_positions', 'end_positions'])
    for feature in ['question_type', 'supp_title_labels', 'supp_sent_labels', 'supp_para_labels']:
        if feature in train_set.features:
            train_set = train_set.remove_columns([feature])

    train_mem_loader = DataLoader(
        train_set,
        shuffle=False,
        collate_fn=default_data_collator,
        batch_size=1,
    )
    # generate mem then add it to the dataset as feature
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=batch_size
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
    model_up.to(device)
    train_loader, model, optim, lr_scheduler = accelerator.prepare(
        train_loader, model, optim, lr_scheduler)
    model_up, train_mem_loader = accelerator.prepare(model_up, train_mem_loader)

    #load saved model lr optim states
    if last_ckpt_num is not None:
        print(f'loading lr, optim, model state from ckpt_{last_ckpt_num}')
        accelerator.load_state(output_dir + f'/ckpt_{last_ckpt_num}')

    # We need to recalculate total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_loader) / gradient_accumulation_steps)
    num_training_steps = num_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate number of training epochs
    num_epochs = math.ceil(num_training_steps / num_update_steps_per_epoch)
    total_num_batches = len(train_loader) // gradient_accumulation_steps
    total_batch_size = batch_size * accelerator.num_processes * gradient_accumulation_steps
    
    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)
    if accelerator.is_local_main_process:
        with open(args.output_dir + '/train_config.json', 'w+') as f:
            json.dump(
                {
                'tokenizer_name': args.tokenizer_name,
                'question_type_num_labels': args.question_type_num_labels,
                'ques_type_loss_weight': args.ques_type_loss_weight,
                'ans_loss_weight': args.ans_loss_weight,
                'para_loss_weight': args.para_loss_weight,
                'sent_loss_weight': args.sent_loss_weight,
                'supp_bce_loss': args.supp_bce_loss,
                'entropy_threshold': args.entropy_threshold,
                'max_mem_len': args.max_mem_len,
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
                'sent_token': SENT_MARKER_END_TOKEN,
                'max_num_answers': args.max_num_answers,
                'max_num_paragraphs': args.max_num_paragraphs,
                'max_num_sentences': args.max_num_sentences,
                'max_seq_len': MAX_SEQ_LEN
            }, f, indent=2)

    num_batches = 0
    for epoch in range(start_epoch, num_epochs):
        model.train()

        accelerator.print('Memory generation started')
        if not os.path.exists(args.output_dir + f'/mem_train_dataset_epoch_{epoch}'):
            if not os.path.exists(args.output_dir + f'/mem_list_epoch_{epoch}.pkl'):
                mem_list, entropy_percentiles_list = generate_full_train_memory(
                    accelerator, model_up, train_mem_loader,
                    paragraph_marker_token=PARA_TOKEN,
                    sentence_marker_token=SENT_MARKER_END_token,
                    entropy_threshold=args.entropy_threshold, max_mem_len=args.max_mem_len,
                ) 
                mem_list = mem_list[:len(train_dataset)]
                entropy_percentiles_list = entropy_percentiles_list[:len(train_dataset)]
                torch.save(mem_list, args.output_dir + f'/mem_list_epoch_{epoch}.pkl')
                torch.save(entropy_percentiles_list, args.output_dir + f'/entropy_percentiles_list_epoch_{epoch}.pkl')
            else:
                mem_list = torch.load(args.output_dir + f'/mem_list_epoch_{epoch}.pkl')
            if accelerator.is_local_main_process:
                list_train_dataset = torch.load(args.raw_train_dataset_path)

                accelerator.print('Memory generation finished')
                ozy = list_train_dataset[:]
                ozy.update({'mem_tokens': mem_list})
                mem_train_dataset = Dataset.from_dict(ozy)
                # build train dataset with memory feature

                # add stride between document segments
                stride = 20
                eos = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

                def preprocess_roberta_long_training_examples(train_examples):
                    questions = [q.strip() for q in train_examples["question"]]
                    inputs = tokenizer(
                        questions,
                        train_examples['context'],
                        max_length=MAX_SEQ_LEN - len(train_examples['mem_tokens'][0]),
                        truncation="only_second",
                        stride=stride,
                        return_overflowing_tokens=True,
                        return_offsets_mapping=True,
                        padding="max_length",
                    )
                    raw_ids = [train_examples['id'][kk] for kk in inputs['overflow_to_sample_mapping']]
                    # store start/end positions of context to filter part of sequence for uncertainty-based topk
                    inputs.update({"id": train_examples['id']})
                    offset_mapping = inputs.pop("offset_mapping")
                    sample_map = inputs.pop("overflow_to_sample_mapping")
                    answers = train_examples['char_answer_offsets']
                    batch_question_type = [] # yes = 0, no = 1, span = 2
                    ex_context_start_id = []
                    ex_context_end_id = []
                    batch_start_positions_list = []
                    batch_end_positions_list = []
                
                    supp_sents = train_examples['supp_sent_char_offsets'] if 'supp_sent_char_offsets' in train_examples.features else None
                    supp_titles = train_examples['supp_title_char_offsets'] if 'supp_title_char_offsets' in train_examples.features else None
                    supp_paras = train_examples['supp_para_char_offsets'] if 'supp_para_char_offsets' in train_examples.features else None


                    batch_title_positions = []
                    batch_sent_positions = []
                    batch_titles_to_sents = []

                    for ex_id in train_examples["id"]:
                        ex_indices = np.where(np.array(raw_ids) == ex_id)[0].tolist()
                        context_token_ids = [np.where(np.array(inputs['input_ids'][i]) == eos)[0] for i in ex_indices]
                        ex_context_start_id.append([elem[1] + 1 for elem in context_token_ids])
                        ex_context_end_id.append([elem[-1] for elem in context_token_ids])
                        start_positions_list = []
                        end_positions_list = []
                        question_type = []
                        supp_sent_start_positions = []
                        supp_sent_end_positions = []
                        supp_title_start_positions = []
                        supp_title_end_positions = []
                        supp_para_start_positions = []

                        for ex_sample in ex_indices:
                            offset_idx = ex_sample
                            offset = offset_mapping[offset_idx]
                            sample_idx = sample_map[offset_idx]
                            sequence_ids = inputs.sequence_ids(offset_idx)
                            # Find the start and end of the context
                            idx = 0
                            while sequence_ids[idx] != 1:
                                idx += 1
                            context_start = idx
                            while sequence_ids[idx] == 1:
                                idx += 1
                            context_end = idx - 1

                            #question type target
                            ques_type_start_char = answers[sample_idx][0][0]
                            ques_type_end_char = answers[sample_idx][0][1]
                            if ques_type_start_char == -1 and ques_type_end_char == -1:
                                question_type.append(0)
                            elif ques_type_start_char == -2 and ques_type_end_char == -2:
                                question_type.append(1)
                            else:
                                question_type.append(2)

                            start_positions = []
                            end_positions = []
                            for answer in answers[sample_idx]:
                                start_char = answer[0]
                                end_char = answer[1]

                                # If the answer is not fully inside the context, label is (0, 0)
                                if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                                    continue
                                else:
                                    idx = context_start
                                    while idx <= context_end and offset[idx][0] <= start_char:
                                        idx += 1
                                    start_positions.append(idx - 1)

                                    idx = context_end
                                    while idx >= context_start and offset[idx][1] >= end_char:
                                        idx -= 1
                                    end_positions.append(idx + 1)
                            start_positions, end_positions = pad_and_drop_duplicates(
                                start_positions, end_positions, args.max_num_answers
                            )
                            start_positions_list.append(start_positions)
                            end_positions_list.append(end_positions)

                            if supp_titles is not None:
                                supp_title_start_positions.append([])
                                supp_title_end_positions.append([])
                                for supp_title_idx in range(len(supp_titles[sample_idx])):
                                    supp_title = supp_titles[sample_idx][supp_title_idx]
                                    supp_title_start_char = supp_title[0]
                                    supp_title_end_char = supp_title[1]
                                    if offset[context_start][0] > supp_title_start_char or \
                                    offset[context_end][1] < supp_title_end_char:
                                        continue
                                    else:
                                        idx1 = context_start
                                        while idx1 <= context_end and offset[idx1][0] <= supp_title_start_char:
                                            idx1 += 1
                                        supp_title_start_positions[-1].append(idx1 - 1)

                                        idx1 = context_end
                                        while idx1 >= context_start and offset[idx1][1] >= supp_title_end_char:
                                            idx1 -= 1
                                        supp_title_end_positions[-1].append(idx1 + 1)
                            if supp_sents is not None:
                                supp_sent_start_positions.append([])
                                supp_sent_end_positions.append([])
                                for supp_sent_idx in range(len(supp_sents[sample_idx])):
                                    supp_sent = supp_sents[sample_idx][supp_sent_idx]
                                    supp_sent_start_char = supp_sent[0]
                                    supp_sent_end_char = supp_sent[1]
                                    if offset[context_start][0] > supp_sent_start_char or \
                                    offset[context_end][1] < supp_sent_end_char:
                                        continue
                                    else:
                                        idx2 = context_start
                                        while idx2 <= context_end and offset[idx2][0] <= supp_sent_start_char:
                                            idx2 += 1
                                        supp_sent_start_positions[-1].append(idx2 - 1)

                                        idx2 = context_end
                                        while idx2 >= context_start and offset[idx2][1] >= supp_sent_end_char:
                                            idx2 -= 1
                                        supp_sent_end_positions[-1].append(idx2 + 1)
                            if supp_paras is not None:
                                supp_para_start_positions.append([])
                                for supp_para_idx in range(len(supp_paras[sample_idx])):
                                    supp_para = supp_paras[sample_idx][supp_para_idx]
                                    supp_para_start_char = supp_para[0]
                                    supp_para_end_char = supp_para[1]
                                    if offset[context_start][0] > supp_para_start_char or offset[context_end][1] < supp_para_end_char:
                                        continue
                                    else:
                                        idx1 = context_start
                                        while idx1 <= context_end and offset[idx1][0] <= supp_para_start_char:
                                            idx1 += 1
                                        supp_para_start_positions[-1].append(idx1 - 1)


                        title_positions = []
                        for i, supp in zip(
                            [inputs['input_ids'][ii] for ii in ex_indices],
                            supp_title_start_positions if supp_titles is not None else supp_para_start_positions
                        ):
                            title_positions.append([0 if j not in supp else 1 for j in np.where(
                                np.array(i) == PARA_TOKEN)[0]]
                                                  )
                            title_positions[-1] = pad_and_drop_duplicates(
                                start_positions=title_positions[-1],
                                max_num_answers=args.max_num_paragraphs
                            )
                        sent_positions = []
                        if supp_sents is not None:
                            for i, supp in zip(
                                [inputs['input_ids'][ii] for ii in ex_indices],
                                supp_sent_end_positions
                            ):
                                sent_positions.append([0 if j not in supp else 1 for j in np.where(np.array(i) == SENT_MARKER_END_token)[0]])
                                sent_positions[-1] = pad_and_drop_duplicates(
                                    start_positions=sent_positions[-1],
                                    max_num_answers=args.max_num_sentences
                                )
                        batch_question_type.append(question_type)
                        batch_start_positions_list.append(start_positions_list)
                        batch_end_positions_list.append(end_positions_list)
                        batch_title_positions.append(title_positions)
                        batch_sent_positions.append(sent_positions)

                    inputs.update({"start_positions": batch_start_positions_list})
                    inputs.update({"end_positions": batch_end_positions_list})
                    inputs.update({'question_type': batch_question_type})
                    inputs.update({'context_start_id': ex_context_start_id})
                    inputs.update({'context_end_id': ex_context_end_id})
                    if supp_titles is not None:
                        inputs.update({"supp_title_labels": batch_title_positions})
                    if supp_sents is not None:
                        inputs.update({"supp_sent_labels": batch_sent_positions})
                    if supp_paras is not None:
                        inputs.update({"supp_para_labels": batch_title_positions})
                    
                    rearranged_inps = []
                    rearranged_masks = []
                    for ex_id in train_examples["id"]:
                        ex_indices = np.where(np.array(raw_ids) == ex_id)[0].tolist()
                        rearranged_inps.append([inputs['input_ids'][i] for i in ex_indices])
                        rearranged_masks.append([inputs['attention_mask'][i] for i in ex_indices])
                    inputs.update({'input_ids': rearranged_inps})
                    inputs.update({'attention_mask': rearranged_masks})
                    return inputs

                mem_train_dataset = mem_train_dataset.map(
                    preprocess_roberta_long_training_examples,
                    batched=True,
                    batch_size=1,
                    remove_columns=mem_train_dataset.column_names,
                )
                ozy = mem_train_dataset[:]
                ozy.update({'mem_tokens': mem_list})
                mem_train_dataset = Dataset.from_dict(ozy)
                mem_train_dataset.save_to_disk(args.output_dir + f'/mem_train_dataset_epoch_{epoch}')
                accelerator.print('Memory-augmented dataset saved')

        accelerator.wait_for_everyone()  
        accelerator.print(f'Loading mem_train_dataset_epoch_{epoch}')
        mem_train_dataset = load_from_disk(args.output_dir + f'/mem_train_dataset_epoch_{epoch}')
        mem_train_dataset = mem_train_dataset.remove_columns(['id'])
        mem_train_loader = DataLoader(
        mem_train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=batch_size,
        )
        mem_train_loader = accelerator.prepare(mem_train_loader)

        for batch in mem_train_loader:
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
                optim.zero_grad(
            if accelerator.sync_gradients:
                progress_bar.update(1)
                num_batches += 1
        print(f"Epoch {epoch} ended, saving the ckpt")
        
        # Save and upload
        # create subfolder for each epoch ckpt related files
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

        print(f'Loading fresh model_up from the latest checkpoint {curr_ckpt_path}')
        model_up = RobertaGEMFormer.from_pretrained(curr_ckpt_path, 
                                                    paragraph_marker_token=PARA_TOKEN, 
                                                    sentence_marker_token=SENT_MARKER_END_token,
                                                    question_type_num_labels=args.question_type_num_labels,
                                                    ques_type_loss_weight=args.ques_type_loss_weight, 
                                                    ans_loss_weight=args.ans_loss_weight,
                                                    para_loss_weight=args.para_loss_weight, 
                                                    sent_loss_weight=args.sent_loss_weight,
                                                    supp_bce_loss=args.supp_bce_loss,
                                                    ans_ce_loss=args.ans_ce_loss
                                                    )
        model_up.resize_token_embeddings(len(tokenizer))
        model_up.to(device)
        model_up = accelerator.prepare(model_up)
        model_up.eval()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
