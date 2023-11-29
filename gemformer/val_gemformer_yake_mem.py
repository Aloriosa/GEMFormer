import argparse
import os
import json
import importlib
import torch
import evaluate
import numpy as np
from tqdm import tqdm
from datetime import datetime
from transformers import RobertaTokenizerFast, default_data_collator
from datasets import load_from_disk 
from datasets.arrow_dataset import Dataset
from accelerate import Accelerator
from torch.utils.data import DataLoader
from modeling_gemformer import RobertaGEMFormer
from utils import compute_metrics, compute_metrics_musique


parser = argparse.ArgumentParser(description="validation")
parser.add_argument("--experiment_dir", type=str, default='../yake_mem_hotpotqa')
parser.add_argument("--val_dataset_path", type=str, default='../hotpotqa_yake_mem_val_dataset')
parser.add_argument("--raw_val_dataset_path", type=str, default='../hotpotqa_val_examples_with_special_seps.pkl')
parser.add_argument("--get_top_2_sp_para", type=bool, default=True)


def main(args):
    
    train_script = 'train_gemformer_yake_mem.py'
    with open(args.experiment_dir+'/train_config.json', 'r') as f:
        train_config = json.load(f)#[-1]
    train_module = importlib.import_module(train_script.split('.py')[0])

    batch_size = 1 
    gradient_accumulation_steps = 1
    tokenizer_name = train_config['tokenizer_name']
    QUESTION_TYPE_NUM_LABELS = train_config['question_type_num_labels']
    QUES_TYPE_LOSS_WEIGHT = train_config['ques_type_loss_weight']
    ANS_LOSS_WEIGHT = train_config['ans_loss_weight']
    PARA_LOSS_WEIGHT = train_config['para_loss_weight']
    SENT_LOSS_WEIGHT = train_config['sent_loss_weight']
    SUPP_BCE_LOSS = train_config.get('supp_bce_loss', None)
    PARA_TOKEN = train_config['para_token']
    SENT_MARKER_END_TOKEN = train_config['sent_token']
    

    validation_dataset = load_from_disk(args.val_dataset_path)
    validation_set = validation_dataset.remove_columns(["example_ids", "offset_mapping"])
    if "titles_to_sents" in validation_set.features:
        validation_set = validation_set.remove_columns(["titles_to_sents"])
    raw_dataset_validation = torch.load(args.raw_val_dataset_path)
    val_loader = DataLoader(validation_set, collate_fn=default_data_collator, 
                            batch_size=batch_size, shuffle=False)

    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    last_ckpt_num = max([int(name.split('_')[-1]) for name in os.listdir(args.experiment_dir) if name.startswith('ckpt_') and os.path.isdir(args.experiment_dir + '/' + name)]) if 'ckpt_' in ' '.join(os.listdir(args.experiment_dir)) else 0
    for curr_val_ckpt in range(train_config['num_epochs']):
        output_dir = args.experiment_dir + f'/ckpt_{curr_val_ckpt}'
        print(f'waiting for ckpt {curr_val_ckpt}')
        while not os.path.exists(output_dir + '/pytorch_model.bin'):
            time.sleep(10)
        print(f'ckpt {curr_val_ckpt} exists!')
        time.sleep(10)

        tokenizer = RobertaTokenizerFast.from_pretrained(output_dir)

        val_log = args.experiment_dir+'/val_log.txt'

        if os.path.exists(output_dir + '/pytorch_model.bin'):
            model_name = output_dir
            print(f"Loading checkpoint from {output_dir.split('/')[-1]}")
        else:
            model_name = tokenizer_name
        assert model_name != tokenizer_name
        print(f"\n\nmodel_name = {model_name}\n\n")

        model = RobertaGEMFormer.from_pretrained(model_name,
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

        device = accelerator.device
        model.to(device)

        val_loader, model = accelerator.prepare(val_loader, model)
        local_output_dir = output_dir+'/analyze_logits'
        if not os.path.exists(local_output_dir):
            os.makedirs(local_output_dir)

        model.eval()
        question_type_logits = []
        start_logits = []
        end_logits = []
        paragraph_logits = []
        sentence_logits = []
        accelerator.print("Evaluation!")

        for batch in tqdm(val_loader):
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
                                      validation_dataset, raw_dataset_validation,
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
