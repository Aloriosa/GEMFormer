import os
import json
import torch

from tqdm import tqdm
from .musique_metrics import compute_metrics as compute_metrics_musique
from .musique_metrics import get_best_span

# list of punctuation tokens from Huggingface RoBERTa-base vocabulary
ROBERTA_BASE_SPECIAL_TOKENS = [1, 2, 4, 6, 12, 22, 35, 36, 43, 60, 72, 93, 108, 111, 113, 116, 126, 128, 131, 
          238, 322, 479, 480, 482, 578, 845, 955, 1215, 1297, 1358, 1589, 1592, 1598, 1640, 1666, 1721, 1917, 
          2156, 2165, 2383, 2652, 2901, 3226, 3256, 3358, 3934, 4332, 4397, 4805, 4832, 4839, 5579, 6600, 6697,
          7479, 7586, 7862, 8061, 8070, 8174, 8488, 8871, 9376, 9957, 10068, 10076, 10116, 10431, 10559, 11227,
          11665, 11888, 12345, 12606, 12651, 12801, 12846, 13198, 13278, 13361, 13373, 13540, 13864, 14025, 
          14220, 14434, 15057, 15483, 15611, 15698, 16276, 16506, 16844, 16998, 17220, 17495, 17516, 17523,
          17809, 18134, 18242, 18456, 18653, 19207, 19246, 19281, 19651, 20186, 20551, 21154, 21277, 21394, 
          21509, 21594, 21704, 21838, 22560, 22896, 23500, 23528, 23962, 23985, 24095, 24303, 24337, 24464, 
          24521, 24524, 24681, 24922, 24965, 24992, 25333, 25522, 25606, 25718, 26487, 26610, 26638, 27079, 
          27144, 27148, 27203, 27223, 27282, 27645, 27785, 27868, 28114, 28553, 28578, 28696, 28749, 28784, 
          29064, 29462, 29482, 29483, 29942, 30115, 30171, 30529, 30550, 30697, 30787, 30831, 31051, 31095, 
          31175, 31274, 31311, 31509, 31558, 31698, 31897, 32269, 32376, 32801, 32965, 33031, 33483, 33525, 
          33647, 34124, 34133, 34199, 34437, 35122, 35227, 35290, 35347, 35524, 35547, 35661, 35965, 36098, 
          36137, 36185, 36380, 36418, 36440, 36467, 36538, 36592, 36738, 36856, 36917, 37008, 37249, 37398, 
          37421, 37637, 37640, 38203, 38304, 38502, 38581, 38713, 38844, 38917, 38947, 39058, 39365, 39550, 
          39574, 39732, 39747, 40021, 40255, 40321, 40323, 40389, 40398, 40635, 40862, 41006, 41039, 41066, 
          41110, 41137, 41478, 41552, 41657, 41667, 41734, 41758, 41833, 41945, 42053, 42078, 42199, 42202, 
          42248, 42254, 42255, 42296, 42326, 42514, 42593, 42604, 42645, 42648, 42654, 42760, 42777, 43002, 
          43003, 43012, 43048, 43074, 43080, 43101, 43303, 43305, 43344, 43353, 43401, 43476, 43521, 43564, 
          43636, 43754, 43775, 43796, 43809, 43839, 43912, 43988, 44065, 44082, 44116, 44162, 44226, 44259, 
          44294, 44371, 44374, 44403, 44408, 44418, 44431, 44440, 44447, 44460, 44516, 44612, 44626, 44629, 
          44660, 44688, 44690, 44706, 44717, 44757, 44832, 44926, 44942, 45056, 45072, 45152, 45177, 45333, 
          45364, 45376, 45381, 45390, 45393, 45405, 45406, 45437, 45587, 45592, 45610, 45627, 45693, 45737, 
          45751, 45793, 45803, 45863, 45894, 45912, 45946, 45973, 46077, 46082, 46117, 46142, 46150, 46156, 
          46161, 46225, 46250, 46253, 46294, 46303, 46343, 46353, 46469, 46479, 46481, 46495, 46564, 46580, 
          46613, 46671, 46679, 46686, 46844, 46904, 46934, 46939, 46961, 46992, 47006, 47033, 47038, 47052, 
          47075, 47096, 47110, 47148, 47155, 47161, 47162, 47259, 47365, 47385, 47426, 47429, 47457, 47460, 
          47517, 47529, 47539, 47567, 47570, 47579, 47619, 47620, 47639, 47655, 47659, 47720, 47770, 47771, 
          47789, 47793, 47813, 47826, 47919, 47965, 48004, 48030, 48037, 48077, 48082, 48086, 48110, 48119, 
          48124, 48134, 48149, 48182, 48188, 48200, 48203, 48209, 48229, 48232, 48256, 48268, 48289, 48292, 
          48298, 48306, 48329, 48336, 48342, 48347, 48364, 48371, 48377, 48404, 48433, 48443, 48457, 48461, 
          48462, 48474, 48505, 48512, 48513, 48520, 48546, 48554, 48562, 48565, 48601, 48610, 48614, 48615, 
          48630, 48634, 48640, 48651, 48654, 48660, 48677, 48691, 48694, 48709, 48712, 48729, 48742, 48749, 
          48752, 48755, 48759, 48771, 48784, 48789, 48794, 48803, 48805, 48817, 48832, 48833, 48835, 48844, 
          48855, 48872, 48880, 48893, 48898, 48900, 48902, 48906, 48919, 48936, 48937, 48948, 48950, 48982, 
          48989, 48999, 49000, 49007, 49024, 49038, 49051, 49058, 49069, 49070, 49071, 49085, 49087, 49092, 
          49095, 49097, 49104, 49123, 49128, 49130, 49138, 49143, 49145, 49151, 49153, 49170, 49177, 49183, 
          49189, 49193, 49196, 49197, 49198, 49201, 49213, 49215, 49216, 49230, 49242, 49248, 49255, 49275, 
          49279, 49281, 49283, 49291, 49293, 49296, 49308, 49314, 49316, 49318, 49319, 49329, 49333, 49338, 
          49346, 49358, 49364, 49366, 49374, 49380, 49384, 49389, 49394, 49410, 49423, 49424, 49434, 49436, 
          49440, 49445, 49452, 49453, 49455, 49463, 49487, 49489, 49509, 49515, 49518, 49521, 49525, 49526, 
          49536, 49563, 49570, 49599, 49604, 49608, 49609, 49612, 49614, 49625, 49629, 49639, 49643, 49655, 
          49666, 49667, 49670, 49674, 49675, 49681, 49688, 49690, 49698, 49701, 49703, 49710, 49712, 49713, 
          49721, 49727, 49731, 49738, 49739, 49747, 49750, 49755, 49761, 49763, 49778, 49783, 49784, 49789, 
          49790, 49795, 49798, 49799, 49800, 49803, 49806, 49812, 49814, 49817, 49826, 49828, 49830, 49836, 
          49849, 49852, 49853, 49858, 49859, 49871, 49882, 49888, 49890, 49893, 49895, 49900, 49903, 49905, 
          49908, 49909, 49910, 49918, 49921, 49923, 49925, 49938, 49940, 49953, 49954, 49959, 49962, 49969, 
          49979, 49982, 49987, 49988, 49991, 49995, 50000, 50003, 50004, 50007, 50012, 50014, 50015, 50016,
          50017, 50018, 50019, 50020, 50024, 50025, 50028, 50031, 50037, 50061, 50065, 50068, 50072, 50078, 
          50084, 50088, 50154, 50155, 50161, 50179, 50184, 50185, 50189, 50193, 50206, 50236, 50254, 50255, 50268]


def token_is_not_content(token):
    """
    Check if a token is a content word
    """
    from rake_nltk import Rake
    r = Rake()
    r.extract_keywords_from_text(token)
    result = r.get_ranked_phrases_with_scores()
    if len(result) == 0:
        return True
    else:
        return False


def create_not_content_words_list(tokenizer):
    decoded_vocab = [tokenizer.decode([i]) for i in tqdm(range(len(tokenizer)))]
    not_content_word_mask = torch.tensor(list(map(token_is_not_content, decoded_vocab)))
    torch.save(not_content_word_mask, 'not_content_word_mask_50270_roberta_vocab.pkl')


if os.path.exists('./not_content_word_mask_50270_roberta_vocab.pkl'):
    NOT_CONTENT_WORD_VOCAB = torch.load('./not_content_word_mask_50270_roberta_vocab.pkl')
else:
    NOT_CONTENT_WORD_VOCAB = []

    
def pad_and_drop_duplicates(start_positions, end_positions=None, max_num_answers=None):
    if end_positions is not None:
        assert len(start_positions) == len(end_positions)
    start_positions = start_positions[:max_num_answers]
    padding_len = max_num_answers - len(start_positions)
    start_positions.extend([-1] * padding_len)

    if end_positions is not None:
        end_positions = end_positions[:max_num_answers]
        end_positions.extend([-1] * padding_len)
        # replace duplicate start/end positions with `-1` because duplicates can result into -ve loss values
        found_start_positions = set()
        found_end_positions = set()
        for i, (start_position, end_position) in enumerate(
            zip(start_positions, end_positions)
        ):
            if start_position in found_start_positions:
                start_positions[i] = -1
            if end_position in found_end_positions:
                end_positions[i] = -1
            found_start_positions.add(start_position)
            found_end_positions.add(end_position)
        return start_positions, end_positions
    else:
        return start_positions    


def add_qa_evidence_tokens(tokenizer,
                           tokens_to_add=[
                           '[question]', '[/question]',
                           '<t>', '</t>', '[/sent]'
                           ]):

    print(f'initial vocab len = {len(tokenizer)}')
    num_added_toks = tokenizer.add_tokens(tokens_to_add)

    print("We have added", num_added_toks, "tokens")
    print(f'final vocab len = {len(tokenizer)}')
    return tokenizer


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    return em, prec, recall

def update_sp(metrics, prediction, gold):
    cur_sp_pred = set(prediction) #set(map(tuple, prediction))
    gold_sp_pred = set(gold) #set(map(tuple, gold))
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    metrics['sp_em'] += em
    metrics['sp_f1'] += f1
    metrics['sp_prec'] += prec
    metrics['sp_recall'] += recall
    return em, prec, recall

def eval_metrics(prediction, gold):
    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
        'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
        'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0}
    for dp in gold:
        cur_id = dp['id']
        can_eval_joint = True
        if cur_id not in prediction['answer']:
            print('missing answer {}'.format(cur_id))
            can_eval_joint = False
        else:
            em, prec, recall = update_answer(
                metrics, prediction['answer'][cur_id], dp['answer'])
        
        if prediction['sp']:
            
            if cur_id not in prediction['sp']:
                print('missing sp fact {}'.format(cur_id))
                can_eval_joint = False
            else:
                sp_em, sp_prec, sp_recall = update_sp(
                    metrics, prediction['sp'][cur_id], dp['supporting_facts'])
        else:
            can_eval_joint = False
            

        if can_eval_joint:
            joint_prec = prec * sp_prec
            joint_recall = recall * sp_recall
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = 0.
            joint_em = em * sp_em

            metrics['joint_em'] += joint_em
            metrics['joint_f1'] += joint_f1
            metrics['joint_prec'] += joint_prec
            metrics['joint_recall'] += joint_recall

    N = len(gold)
    for k in metrics.keys():
        metrics[k] /= N
    return metrics



def compute_metrics(question_type_logits, start_logits, end_logits,
                    paragraph_logits, sentence_logits,
                    features, examples, output_dir=None,
                    supp_bce_loss=None, get_top_2_sp_para=True):
    n_best = 20
    max_answer_length = 30
    sp_threshold = 0.3
    question_types = np.argmax(question_type_logits, 1)
    predict_para_support_np = []

    if paragraph_logits is not None:
        if supp_bce_loss == True:
            if len(paragraph_logits[0].shape) == 3 and paragraph_logits[0].shape[-1] == 1:
                paragraph_logits = [np.squeeze(np.array(i),axis=-1).tolist() for i in paragraph_logits]
                for para_log in tqdm(paragraph_logits):
                    lst = []
                    for i in para_log:
                      lst.append([])
                      for j in i:
                        if j != -np.inf:
                            try: 
                                lst[-1].append(j)
                            except:
                                print(f"lst {lst}, j {j}")
                    predict_para_support_np.append(lst)
            else:
                predict_para_support_np = paragraph_logits
                predict_para_support_np = [
                torch.sigmoid(torch.tensor(i)).data.cpu().numpy().tolist() for i in predict_para_support_np]

        else:
            for para_log in tqdm(paragraph_logits):
                lst = []
                for i, sigmoid_i in zip(para_log,
                                        torch.nn.functional.softmax(torch.tensor(para_log),
                                                                    dim=-1).data.cpu().numpy()):
                  lst.append([])
                  for j, sigmoid_j in zip(i, sigmoid_i):
                    if all(j != np.array([-np.inf, -np.inf])):
                        try: 
                            lst[-1].append(sigmoid_j[1])
                        except:
                            print(f"lst {lst}, sigmoid_j {sigmoid_j}")
                predict_para_support_np.append(lst)

    predict_sent_support_np = []
    if sentence_logits is not None:
        if supp_bce_loss == True:
            if len(sentence_logits[0].shape) == 3 and sentence_logits[0].shape[-1] == 1:
                sentence_logits = [np.squeeze(np.array(i),axis=-1).tolist() for i in sentence_logits]

                for sent_log in tqdm(sentence_logits):
                    lst = []
                    for i in sent_log:
                      lst.append([])
                      for j in i:
                        if j != -np.inf:
                            try: 
                                lst[-1].append(j)
                            except:
                                print(f"lst {lst}, j {j}")
                    predict_sent_support_np.append(lst)
            else:
                predict_sent_support_np = sentence_logits
                predict_sent_support_np = [
                torch.sigmoid(torch.tensor(i)).data.cpu().numpy().tolist() for i in predict_sent_support_np]
        else:
            for sent_log in tqdm(sentence_logits):
                lst = []
                for i, sigmoid_i in zip(sent_log,
                                        torch.nn.functional.softmax(torch.tensor(sent_log),
                                                                    dim=-1).data.cpu().numpy()):
                  lst.append([])
                  for j, sigmoid_j in zip(i, sigmoid_i):
                    if all(j != np.array([-np.inf, -np.inf])):
                        try: 
                            lst[-1].append(sigmoid_j[1])
                        except:
                            print(f"lst {lst}, sigmoid_j {sigmoid_j}")
                predict_sent_support_np.append(lst)

    predicted_answers = {}
    sp_para_dict = {}
    sp_sent_dict = {}

    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        if len(features) > 1:
            feature_index = features["example_ids"].index(example_id)
        else:
            feature_index = 0

        question_type = question_types[feature_index]
        if question_type == 0:
            predicted_answers[str(example_id)] = "yes"
        elif question_type == 1:
            predicted_answers[str(example_id)] = "no"
        elif question_type == 2:
            for i in range(len(start_logits[feature_index])):
                start_logit = start_logits[feature_index][i]
                end_logit = end_logits[feature_index][i]
                offsets = features[feature_index]["offset_mapping"][i]

                start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
                end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Skip answers that are not fully in the context
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        # Skip answers with a length that is either < 0 or > max_answer_length
                        if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                        ):
                            continue

                        answer = {
                            "text": context[offsets[start_index][0] : offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                            'chunk_idx': i
                        }
                        answers.append(answer)

            # Select the answer with the best score
            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x["logit_score"])
                predicted_answers[str(example_id)] = best_answer["text"]
                predicted_answers[str(example_id)+'_chunk_idx'] = best_answer['chunk_idx']
            else:
                predicted_answers[str(example_id)] = ""
        else:
            assert False

        if paragraph_logits is not None:
            #get top 2 paragraphs
            if supp_bce_loss == True:
                concat_predict_para_support_np = predict_para_support_np[feature_index]
            else:
                concat_predict_para_support_np = []
                for i in predict_para_support_np[feature_index]:
                    concat_predict_para_support_np += i

            para_pred_ids_raw = np.argsort(concat_predict_para_support_np)[-1:-2-1:-1] \
            if (get_top_2_sp_para == True) else np.arange(len(concat_predict_para_support_np))
            para_pred_ids = np.array(features[feature_index]["concat_titles_ids"])[para_pred_ids_raw]

            sp_para_dict.update({str(example_id): para_pred_ids.tolist()})

            titles_to_sents = features[feature_index]["titles_to_sents"]

            if supp_bce_loss == True:
                concat_titles_to_sents = []
                for chunk in titles_to_sents:
                    for _, i in enumerate(chunk):
                        if len(i) > 0:
                            concat_titles_to_sents += [[_, j[0],j[1]] for j in i]
                concat_titles_to_sents = [[j[0],ii,j[2]] for ii, j in enumerate(concat_titles_to_sents)]
                cur_sent_sp_pred = []
                for j in para_pred_ids:
                    for sent_id in [i[1:] for i in concat_titles_to_sents if i[0] == j]:
                        if predict_sent_support_np[feature_index][sent_id[0]] > sp_threshold:
                            cur_sent_sp_pred.append(sent_id[-1])
                cur_sent_sp_pred = np.unique(sorted(cur_sent_sp_pred)).tolist()
            else:
                cur_sent_sp_pred = []
                for j in para_pred_ids:
                    for chunk, local_global_sent_ids_list in enumerate([i[j] for i in titles_to_sents]):
                        for sent_id in local_global_sent_ids_list:
                            if predict_sent_support_np[feature_index][chunk][sent_id[0]] > sp_threshold:
                                cur_sent_sp_pred.append(sent_id[-1])
                cur_sent_sp_pred = np.unique(sorted(cur_sent_sp_pred)).tolist()

            sp_sent_dict.update({str(example_id): cur_sent_sp_pred})

    gold = [{"id": str(ex["id"]),  
                            "answer": ex["answer"],
                            'sp_para': ex["supp_title_ids"],
                            'supporting_facts': ex["supp_sent_ids"]
                           } for ex in examples]
    prediction = {'answer': predicted_answers, 'sp_para': sp_para_dict, 'sp': sp_sent_dict}
    if output_dir is not None:
        with open(output_dir + '/golden_targets.json', 'w') as f:
            json.dump(gold, f)
        with open(output_dir + '/predictions.json', 'w') as f:
            json.dump(prediction, f)

    return eval_metrics(prediction, gold)


def compute_metrics_musique(start_logits, end_logits, paragraph_logits,
                            features, examples, output_dir=None, supp_bce_loss=None
                            ):
    n_best = 20
    max_answer_length = 30
    predict_para_support_np = []

    if paragraph_logits is not None:
        if supp_bce_loss == True:
            if len(paragraph_logits[0].shape) == 3 and paragraph_logits[0].shape[-1] == 1:
                paragraph_logits = [np.squeeze(np.array(i),axis=-1).tolist() for i in paragraph_logits]
                for para_log in tqdm(paragraph_logits):
                    lst = []
                    for i in para_log:
                      lst.append([])
                      for j in i:
                        if j != -np.inf:
                            try: 
                                lst[-1].append(j)
                            except:
                                print(f"lst {lst}, j {j}")
                    predict_para_support_np.append(lst)
            else:
                predict_para_support_np = [np.squeeze(np.array(i),axis=-1).tolist() for i in paragraph_logits]
                predict_para_support_np = [torch.sigmoid(torch.tensor(i)).data.cpu().numpy().tolist() \
                for i in predict_para_support_np]

        else:
            for para_log in tqdm(paragraph_logits):
                lst = []
                for i, sigmoid_i in zip(para_log,
                                        torch.nn.functional.softmax(torch.tensor(para_log),
                                                                    dim=-1).data.cpu().numpy()):
                  lst.append([])
                  for j, sigmoid_j in zip(i, sigmoid_i):
                    if all(j != np.array([-np.inf, -np.inf])):
                        try: 
                            lst[-1].append(sigmoid_j[1])
                        except:
                            print(f"lst {lst}, sigmoid_j {sigmoid_j}")
                predict_para_support_np.append(lst)

    sp_para_dict = {}
    predictions = []
    for example in tqdm(examples):
        prediction = {}
        example_id = example["id"]
        prediction.update({'id': example["id"]})
        context = example["context"]
        answers = []
        feature_index = features["example_ids"].index(example_id)
        for i in range(len(start_logits[feature_index])):
            start_logit = start_logits[feature_index][i]
            end_logit = end_logits[feature_index][i]
            offsets = features[feature_index]["offset_mapping"][i]

            for no_mem_sep_cntxt_start, i in enumerate(offsets):
                if i != None:
                    break
            no_mem_sep_cntxt_start += 2
            
            best_span, best_span_score = get_best_span(torch.tensor([start_logit.tolist()[no_mem_sep_cntxt_start:]]),
                                                       torch.tensor([end_logit.tolist()[no_mem_sep_cntxt_start:]]),
                                                       max_length=None)
            start_index = best_span[0] + no_mem_sep_cntxt_start
            end_index = best_span[1] + no_mem_sep_cntxt_start
            
            if offsets[start_index] is None or offsets[end_index] is None:
                continue
            # Skip answers with a length that is either < 0 or > max_answer_length
            if (end_index < start_index):
                continue
            answer = {
                "text": context[offsets[start_index][0] : offsets[end_index][1]],
                "logit_score": best_span_score,
                'chunk_idx': i
            }
            answers.append(answer)

        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            prediction.update({'predicted_answer': best_answer["text"]})
        else:
            prediction.update({'predicted_answer': ''})
        num_supp_paras = len(example["supp_para_ids"])
        if paragraph_logits is not None:
            #get top 2 paragraphs
            if supp_bce_loss == True:
                concat_predict_para_support_np = predict_para_support_np[feature_index]
            else:
                concat_predict_para_support_np = []
                for i in predict_para_support_np[feature_index]:
                    concat_predict_para_support_np += i

            para_pred_ids_raw = np.argsort(concat_predict_para_support_np)[-1 : -num_supp_paras - 1 : -1]
            para_pred_ids = np.array(features[feature_index]["concat_titles_ids"])[para_pred_ids_raw]
            prediction.update({'predicted_support_idxs': para_pred_ids.tolist()})
        predictions.append(prediction)

    gold = [{"id": str(ex["id"]),  
                            "answer": ex["answer"],
                            'answer_aliases': ex['answer_aliases'],
                            'supp_para_ids': ex["supp_para_ids"],
                           } for ex in examples]

    if output_dir is not None:
        with open(os.path.join(output_dir, 'golden_targets.json'), 'w') as f:
            json.dump(gold, f)
        with open(os.path.join(output_dir, 'predictions.json'), 'w') as f:
            json.dump(predictions, f)

    return compute_metrics_musique(predictions, gold)
