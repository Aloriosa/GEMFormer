import random
from tqdm import tqdm
import spacy
import ujson as json
from collections import Counter
import numpy as np
import os.path
import argparse
import torch
import torch
import os
from joblib import Parallel, delayed

import torch

nlp = spacy.blank("en")

import bisect
import re


def find_nearest(a, target, test_func=lambda x: True):
    idx = bisect.bisect_left(a, target)
    if (0 <= idx < len(a)) and a[idx] == target:
        return target, 0
    elif idx == 0:
        return a[0], abs(a[0] - target)
    elif idx == len(a):
        return a[-1], abs(a[-1] - target)
    else:
        d1 = abs(a[idx] - target) if test_func(a[idx]) else 1e200
        d2 = abs(a[idx-1] - target) if test_func(a[idx-1]) else 1e200
        if d1 > d2:
            return a[idx-1], d2
        else:
            return a[idx], d1

        
def fix_span(para, span):
    span = span.strip()
    parastr = "".join(para)
    best_indices = []
    if span in parastr:
        if span == parastr:
            return (0, len(parastr))

        for m in re.finditer(re.escape(span), parastr):
            begin_offset, end_offset = m.span()

            best_indices.append([begin_offset, end_offset])

    return best_indices


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        token = token.strip()# for tokens with right and left spaces that not separated by 2 spaces in the text
        pre = current
        current = text.find(token, current)
        if current < 0:
            print(f"{token} not found in {text}")
            raise Exception(f"{token} not found in\n{text}\ntokens = {tokens}\npre {pre}")
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def prepro_sent(sent):
    return sent


def get_start_end(text, text_context):
    if text.strip() not in ''.join(text_context):
        best_indices = [[0, 0]]
    else:
        best_indices = fix_span(text_context, text.strip())
        
    return best_indices


def _process_article(article, with_special_seps=False):
    question_start_sep = ''
    question_end_sep = ''
    title_start_sep = ''
    title_end_sep = ''
    sent_end_sep = ''
    if with_special_seps:
        #[question]q q q[/question]<t>title 1</t>s_11 s_12 s_13[/sent]s_21 s_22 s_23[/sent]...<t>title 2</t>s_21 s_22 s_23[/sent]....[/sent]

        question_start_sep = ''
        question_end_sep = ''
        para_sep = '[para]'
        
    text_context = ''
    para_titles = article['context']['title']
    para_sentences = article['context']['sentences']
    
    paragraphs = list(zip(para_titles, para_sentences))
    if len(paragraphs) == 0:
        paragraphs = [['some random title', 'some random stuff']]
        
    ques_txt = question_start_sep + article['question'] + question_end_sep
    
    supp_sent_ids = []
    supp_sent_texts = []
    supp_sent_char_offsets = []
    supp_para_ids = []
    supp_title_texts = []
    supp_para_char_offsets = []
    if 'supporting_facts' in article:
        supp_title_texts = article['supporting_facts']['title']
        supp_para_ids = [para_titles.index(item) for item in supp_title_texts]
    else:
        supp_para_ids = []
    
    sent_counter = 0
    for para_idx, para in enumerate(paragraphs):
        _, cur_para = para[0], para[1]
        para_context = ''
        for sent_id, sent in enumerate(cur_para):
            if sent_id == 0:
                sent = para_sep + sent
            para_context += sent
            
            sent_counter += 1
        if para_idx in supp_para_ids:
            if with_special_seps:
                supp_para_start = para_context.rfind(para_sep)
                supp_para_char_offsets.append([supp_para_start + len(text_context), 
                                               supp_para_start + len(para_sep) + len(text_context)])

            if 'answer' in article:
                best_indices = [0, 0]
                answer = article['answer'].strip()
                assert answer.lower() not in ['yes', 'no']
                candidate_indices = fix_span(para_context, article['answer'].strip())
                if candidate_indices != []:
                    best_indices = [candidate_indices[0][0] + len(text_context), candidate_indices[0][1] + len(text_context)]
            else:
                # some random stuff
                answer = 'random'
                best_indices = (0, 1)
               
        text_context += para_context
    example = {'context': text_context,
               'question': ques_txt,
               'answer': answer,
               'char_answer_offsets': best_indices, 
               'id': article['id'],
               'supp_para_char_offsets': supp_para_char_offsets,
               'supp_para_ids': supp_para_ids,
               'supp_title_texts': supp_title_texts,
               'answer_aliases': article['answer_aliases']
              }
    
    return example

def process_file(data, with_special_seps=False):
    
    examples = []
    eval_examples = {}
    
    outputs = Parallel(n_jobs=12, verbose=10)(delayed(_process_article)(article, with_special_seps) for article in data)
    print("{} questions in total".format(len(outputs)))

    return outputs