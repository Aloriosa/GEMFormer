import argparse
import os
import json
import importlib
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from transformers import RobertaTokenizerFast, default_data_collator
from datasets import load_from_disk 
from datasets.arrow_dataset import Dataset
from accelerate import Accelerator
from torch.utils.data import DataLoader
from modeling_gemformer import RobertaGEMFormer
from utils import compute_metrics, compute_metrics_musique, pad_and_drop_duplicates


parser = argparse.ArgumentParser(description="validation")
parser.add_argument("--experiment_dir", type=str, default='../gemformer_highest_hotpotqa')
parser.add_argument("--val_dataset_path", type=str, default='../hotpotqa_preprocessed_val_examples_512_multitask_stride20_one_doc_batched_without_zero_answer_pos')
parser.add_argument("--raw_val_dataset_path", type=str, default='../hotpotqa_val_examples_with_special_seps.pkl')
parser.add_argument("--get_top_2_sp_para", type=bool, default=True)


def main(args):
    train_script = 'train_gemformer.py'
    with open(args.experiment_dir+'/train_config.json', 'r') as f:
        train_config = json.load(f)
    train_module = importlib.import_module(train_script.split('.py')[0])

    batch_size = 1 
    gradient_accumulation_steps = 1
    TOKENIZER_NAME = train_config['tokenizer_name']
    MAX_MEM_LEN = train_config['max_mem_len']
    QUESTION_TYPE_NUM_LABELS = train_config['question_type_num_labels']
    QUES_TYPE_LOSS_WEIGHT = train_config['ques_type_loss_weight']
    ANS_LOSS_WEIGHT = train_config['ans_loss_weight']
    PARA_LOSS_WEIGHT = train_config['para_loss_weight']
    SENT_LOSS_WEIGHT = train_config['sent_loss_weight']
    ENTROPY_THRESHOLD = train_config['entropy_threshold']
    PARA_TOKEN = train_config['para_token']
    SENT_MARKER_END_TOKEN = train_config['sent_token']
    SUPP_BCE_LOSS = train_config.get('supp_bce_loss', None)
    MAX_NUM_PARAGRAPHS = train_config['max_num_paragraphs']
    MAX_NUM_SENTENCES = train_config['max_num_sentences']
    MAX_SEQ_LEN = train_config['max_seq_len']

    validation_dataset = load_from_disk(args.val_dataset_path)
    validation_set = validation_dataset.remove_columns(["example_ids", 
                                                        "offset_mapping"
                                                       ])
    if "titles_to_sents" in validation_set.features:
        validation_set = validation_set.remove_columns(["titles_to_sents"])
    raw_dataset_validation = torch.load(args.raw_val_dataset_path)
    val_loader = DataLoader(validation_set, collate_fn=default_data_collator, 
                            batch_size=batch_size, shuffle=False)

    for feature in ['supp_title_labels', 'supp_sent_labels']:
        if feature in validation_set.features:
            validation_set = validation_set.remove_columns([feature])
    val_mem_loader = DataLoader(
        validation_set,
        shuffle=False,
        collate_fn=default_data_collator,
        batch_size=1
    )
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)

    for curr_val_ckpt in range(train_config['num_epochs']):
        output_dir = args.experiment_dir + f'/ckpt_{curr_val_ckpt}'
        print(f'waiting for ckpt {curr_val_ckpt}')
        while not os.path.exists(output_dir + '/pytorch_model.bin'):
            time.sleep(10)
        print(f'ckpt {curr_val_ckpt} exists!')
        time.sleep(10)

        tokenizer = RobertaTokenizerFast.from_pretrained(output_dir)

        val_log = args.experiment_dir + '/val_log.txt'
        if os.path.exists(output_dir + '/pytorch_model.bin'):
            model_name = output_dir
            print(f"Loading checkpoint from {output_dir.split('/')[-1]}")
        else:
            model_name = TOKENIZER_NAME
        assert model_name != TOKENIZER_NAME

        model = RobertaGEMFormer.from_pretrained(
            model_name, 
            paragraph_marker_token=PARA_TOKEN,
            sentence_marker_token=SENT_MARKER_END_TOKEN,
            question_type_num_labels=QUESTION_TYPE_NUM_LABELS,
            ques_type_loss_weight=QUES_TYPE_LOSS_WEIGHT,
            ans_loss_weight=ANS_LOSS_WEIGHT,
            para_loss_weight=PARA_LOSS_WEIGHT,
            sent_loss_weight=SENT_LOSS_WEIGHT,
            supp_bce_loss=SUPP_BCE_LOSS
        )
        model.resize_token_embeddings(len(tokenizer))
        # model for estimation of uncertainty of predictions
        model_up = RobertaGEMFormer.from_pretrained(
            model_name, 
            paragraph_marker_token=PARA_TOKEN,
            sentence_marker_token=SENT_MARKER_END_TOKEN,
            question_type_num_labels=QUESTION_TYPE_NUM_LABELS,
            ques_type_loss_weight=QUES_TYPE_LOSS_WEIGHT,
            ans_loss_weight=ANS_LOSS_WEIGHT,
            para_loss_weight=PARA_LOSS_WEIGHT,
            sent_loss_weight=SENT_LOSS_WEIGHT,
            supp_bce_loss=SUPP_BCE_LOSS
        )
        model_up.resize_token_embeddings(len(tokenizer))

        device = accelerator.device
        model.to(device)
        model_up.to(device)
        val_loader, model = accelerator.prepare(val_loader, model)
        model_up, val_mem_loader = accelerator.prepare(model_up, val_mem_loader)
        
        local_output_dir = output_dir + '/analyze_logits'
        if not os.path.exists(local_output_dir):
            os.makedirs(local_output_dir)

        model.eval()
        question_type_logits = []
        start_logits = []
        end_logits = []
        paragraph_logits = []
        sentence_logits = []
        accelerator.print("Evaluation!")
        mem_list, entropy_percentiles_list = train_module.generate_full_train_memory(
            accelerator, model_up, 
            val_mem_loader,
            paragraph_marker_token=PARA_TOKEN,
            sentence_marker_token=SENT_MARKER_END_TOKEN,
            output_dir=output_dir,
            entropy_threshold=ENTROPY_THRESHOLD,
            max_mem_len=MAX_MEM_LEN
        )
        torch.save(mem_list, local_output_dir + f'/val_mem_list.pkl')
        torch.save(entropy_percentiles_list, local_output_dir + f'/val_entropy_percentiles_list.pkl')

        if accelerator.is_local_main_process:
            list_val_dataset = torch.load(args.raw_val_dataset_path)
            accelerator.print('Memory generation finished')
            list_val_dataset_to_dict = list_val_dataset[:]
            list_val_dataset_to_dict.update({'mem_tokens': mem_list[:len(list_val_dataset)]})
            mem_val_dataset = Dataset.from_dict(list_val_dataset_to_dict)

            stride = 20
            eos = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
            def preprocess_longformer_validation_examples(examples):
                questions = [q.strip() for q in examples["question"]]
                inputs = tokenizer(
                    questions,
                    examples["context"],
                    max_length=MAX_SEQ_LEN - len(examples['mem_tokens'][0]),
                    truncation="only_second",
                    stride=stride,
                    return_overflowing_tokens=True,
                    return_offsets_mapping=True,
                    padding="max_length",
                )
                # store raw sample id to match subsamples related to the same big context document 
                raw_ids = [examples['id'][kk] for kk in inputs['overflow_to_sample_mapping']]
                # store start/end positions of context to filter part of sequence for uncertainty-based topk
                inputs.update({"example_ids": examples['id']})
                offset_mapping = inputs["offset_mapping"]
                sample_map = inputs.pop("overflow_to_sample_mapping")
                ex_context_start_id = []
                ex_context_end_id = []
                supp_sents = examples['supp_sent_char_offsets'] if 'supp_sent_char_offsets' in examples.features else None
                supp_titles = examples['supp_title_char_offsets'] if 'supp_title_char_offsets' in examples.features else None
                supp_paras = examples['supp_para_char_offsets'] if 'supp_para_char_offsets' in examples.features else None

                batch_title_positions = []
                batch_sent_positions = []
                batch_titles_to_sents = []
                batch_concat_titles_ids = []

                for ex_id in examples["id"]:
                    ex_indices = np.where(np.array(raw_ids) == ex_id)[0].tolist()
                    context_token_ids = [np.where(np.array(inputs['input_ids'][i]) == eos)[0] for i in ex_indices]
                    ex_context_start_id.append([elem[1] + 1 for elem in context_token_ids])
                    ex_context_end_id.append([elem[-1] for elem in context_token_ids])

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
                        inputs["offset_mapping"][offset_idx] = [
                            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
                        ]
                        # Find the start and end of the context
                        idx = 0
                        while sequence_ids[idx] != 1:
                            idx += 1
                        context_start = idx
                        while sequence_ids[idx] == 1:
                            idx += 1
                        context_end = idx - 1

                        if supp_titles is not None:
                            supp_title_start_positions.append([])
                            supp_title_end_positions.append([])
                            for supp_title_idx in range(len(supp_titles[sample_idx])):
                                supp_title = supp_titles[sample_idx][supp_title_idx]
                                supp_title_start_char = supp_title[0]
                                supp_title_end_char = supp_title[1]

                                # If the answer is not fully inside the context, label is (0, 0)
                                if offset[context_start][0] > supp_title_start_char or offset[context_end][1] < supp_title_end_char:
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

                                # If the answer is not fully inside the context, label is (0, 0)
                                if offset[context_start][0] > supp_sent_start_char or offset[context_end][1] < supp_sent_end_char:
                                    continue
                                else:
                                    # Otherwise it's the start and end token positions
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
                    for i, supp in zip([inputs['input_ids'][ii] for ii in ex_indices],
                                       supp_title_start_positions if supp_titles is not None else supp_para_start_positions):
                        title_positions.append(
                            [0 if j not in supp else 1 for j in np.where(np.array(i) == PARA_TOKEN)[0]])
                        title_positions[-1] = pad_and_drop_duplicates(
                            start_positions=title_positions[-1],
                            max_num_answers=MAX_NUM_PARAGRAPHS)

                    sent_positions = []
                    if supp_sents is not None:
                        for i, supp in zip([inputs['input_ids'][ii] for ii in ex_indices], supp_sent_end_positions):
                            sent_positions.append(
                                [0 if j not in supp else 1 for j in np.where(np.array(i) == SENT_MARKER_END_TOKEN)[0]])
                            sent_positions[-1] = pad_and_drop_duplicates(
                                start_positions=sent_positions[-1],
                                max_num_answers=MAX_NUM_SENTENCES)

                    concat_input_ids = []
                    concat_offset_mapping = []
                    for ii in ex_indices:
                        concat_input_ids += inputs['input_ids'][ii]
                        concat_offset_mapping += inputs['offset_mapping'][ii]

                    titles = np.where(np.array(concat_input_ids) == PARA_TOKEN)[0]
                    np_titles_concat_offset_mapping = np.array(concat_offset_mapping)[titles]
                    titles_offsets_unique = sorted(np.unique(np_titles_concat_offset_mapping.tolist(), axis=0).tolist())
                    concat_titles_ids = [titles_offsets_unique.index(list(i)) for i in np_titles_concat_offset_mapping]

                    titles_to_sents = []
                    if supp_sents is not None:
                        global_sents = np.where(np.array(concat_input_ids) == SENT_MARKER_END_TOKEN)[0]
                        global_sents_offsets_unique = []
                        for i in np.array(concat_offset_mapping)[global_sents]:
                            if i not in global_sents_offsets_unique:
                                global_sents_offsets_unique.append(i)

                        for i, offsets_list in zip([inputs['input_ids'][ii] for ii in ex_indices],
                                                   [inputs['offset_mapping'][ii] for ii in ex_indices]):
                            sents = np.where(np.array(i) == SENT_MARKER_END_TOKEN)[0].tolist()
                            sents_offsets = np.array(offsets_list)[sents]
                            tmp = []
                            if len(titles_offsets_unique) > 1:
                                #local chunk sent_id, global doc sent_id
                                tmp = [
                                [(sent_id,
                                  global_sents_offsets_unique.index(sent_offset)) for sent_id, \
                                sent_offset in enumerate(sents_offsets) if ((sent_offset[0] >= title_offset[-1]) and \
                                                                            (sent_offset[-1] <= titles_offsets_unique[i+1][0]))] for i, \
                                title_offset in enumerate(titles_offsets_unique[:-1])]

                            if len(titles_offsets_unique) > 0:
                                tmp.append([
                                          (sent_id,
                                           global_sents_offsets_unique.index(sent_offset)) for sent_id, \
                                          sent_offset in enumerate(sents_offsets) if (sent_offset[0] >= titles_offsets_unique[-1][-1])])

                            titles_to_sents.append(tmp)

                    batch_concat_titles_ids.append(concat_titles_ids)
                    batch_title_positions.append(title_positions)
                    if supp_sents is not None:
                        batch_sent_positions.append(sent_positions)
                        batch_titles_to_sents.append(titles_to_sents)
                inputs.update({'context_start_id': ex_context_start_id})
                inputs.update({'context_end_id': ex_context_end_id})
                if supp_titles is not None:
                    inputs.update({"supp_title_labels": batch_title_positions})
                if supp_paras is not None:
                    inputs.update({"supp_para_labels": batch_title_positions})
                if supp_sents is not None:
                    inputs.update({"supp_sent_labels": batch_sent_positions})
                    inputs.update({"titles_to_sents": batch_titles_to_sents})
                inputs.update({"concat_titles_ids": batch_concat_titles_ids})

                rearranged_inps = []
                rearranged_masks = []
                rearranged_offset_mapping = []

                for ex_id in examples["id"]:
                    ex_indices = np.where(np.array(raw_ids) == ex_id)[0].tolist()
                    rearranged_inps.append([inputs['input_ids'][i] for i in ex_indices])
                    rearranged_masks.append([inputs['attention_mask'][i] for i in ex_indices])
                    rearranged_offset_mapping.append([inputs['offset_mapping'][i] for i in ex_indices])
                inputs.update({'input_ids': rearranged_inps})
                inputs.update({'attention_mask': rearranged_masks})
                inputs.update({'offset_mapping': rearranged_offset_mapping})
                return inputs

            mem_val_dataset = mem_val_dataset.map(
                preprocess_longformer_validation_examples,
                batched=True,
                batch_size=1,
                remove_columns=mem_val_dataset.column_names
            )
            mem_val_dataset_upd = mem_val_dataset[:]
            mem_val_dataset_upd.update({'mem_tokens': mem_list[:len(mem_val_dataset)]})
            mem_val_dataset = Dataset.from_dict(mem_val_dataset_upd)
            mem_val_dataset.save_to_disk(local_output_dir + f'/mem_val_dataset')
            accelerator.print('Memory-augmented dataset saved')
        accelerator.wait_for_everyone()  
        accelerator.print('Loading mem_val_dataset')
        mem_val_dataset = load_from_disk(local_output_dir + f'/mem_val_dataset')
        mem_val_set = mem_val_dataset.remove_columns(['example_ids', 'offset_mapping'])
        if 'titles_to_sents' in mem_val_set.features:
            mem_val_set = mem_val_set.remove_columns(['titles_to_sents'])
        mem_val_loader = DataLoader(
            mem_val_set,
            collate_fn=default_data_collator,
            batch_size=batch_size,
        )
        mem_val_loader = accelerator.prepare(mem_val_loader)

        for batch in tqdm(mem_val_loader):
            with torch.no_grad():
                outputs = model(input_ids=batch['input_ids'][0],
                                attention_mask=batch['attention_mask'][0],
                                context_start_id=batch['context_start_id'][0],
                                paragraph_labels=batch['supp_title_labels'][0] if 'supp_title_labels' in batch else batch['supp_para_labels'][0],
                                sentence_labels=batch['supp_sent_labels'][0] if SENT_MARKER_END_TOKEN is not None else None,
                                mem_tokens=batch['mem_tokens'][0],
                                eval_flag=True
                               )
            if QUESTION_TYPE_NUM_LABELS is not None:
                question_type_logits.append(accelerator.gather(outputs.question_type_logits).cpu().numpy())
            if SENT_MARKER_END_TOKEN is not None:
                sentence_logits.append(accelerator.gather(outputs.supp_sentence_logits).cpu().numpy())
            start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
            end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())
            paragraph_logits.append(accelerator.gather(outputs.supp_paragraph_logits).cpu().numpy())

        if QUESTION_TYPE_NUM_LABELS is not None:
            question_type_logits = np.concatenate(question_type_logits)
            question_type_logits = question_type_logits[: len(raw_dataset_validation)]
            torch.save(question_type_logits, output_dir+'/question_type_logits.pkl')
        if SENT_MARKER_END_TOKEN is not None:
            sentence_logits = sentence_logits[: len(raw_dataset_validation)]
            torch.save(sentence_logits, output_dir+'/sentence_logits.pkl')

        start_logits = start_logits[: len(raw_dataset_validation)]
        end_logits = end_logits[: len(raw_dataset_validation)]
        paragraph_logits = paragraph_logits[: len(raw_dataset_validation)]
        torch.save(start_logits, output_dir+'/start_logits.pkl')
        torch.save(end_logits, output_dir+'/end_logits.pkl')
        torch.save(paragraph_logits, output_dir+'/paragraph_logits.pkl')
        
        if QUESTION_TYPE_NUM_LABELS is not None:
            metrics = compute_metrics(question_type_logits, start_logits, end_logits, 
                                      paragraph_logits, sentence_logits, 
                                      mem_val_dataset, raw_dataset_validation,
                                      output_dir=output_dir,
                                      supp_bce_loss=SUPP_BCE_LOSS,
                                      get_top_2_sp_para=args.get_top_2_sp_para
                                     )
        else:
            metrics = compute_metrics_musique(start_logits, end_logits, paragraph_logits,
                                              mem_val_dataset, raw_dataset_validation,
                                              output_dir=output_dir, supp_bce_loss=SUPP_BCE_LOSS
                                              )
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} eval {output_dir.split('/')[-1]}:", metrics)
        f = open(val_log,'a') 
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} eval {output_dir.split('/')[-1]}: {metrics}\n")
        f.close()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
