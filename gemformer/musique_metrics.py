# code adopted from https://github.com/StonyBrookNLP/musique/blob/bec107f5756f067657778ee89a9add6902191e7a/metrics

"""
Answer metric -- mostly taken directly from squad_tools of allennlp.
"""
import re
import torch
import string
import collections
from typing import Tuple, List, Any, Dict

"""
An abstract class representing a metric which can be accumulated.
"""
class Metric:
    """
    An abstract class representing a metric which can be accumulated.
    """

    def __call__(self, predictions: Any, gold_labels: Any):
        raise NotImplementedError

    def get_metric(self, reset: bool) -> Dict[str, Any]:
        """
        Compute and return the metric. Optionally also call `self.reset`.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Reset any accumulators or internal state.
        """
        raise NotImplementedError


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


class AnswerMetric(Metric):
    def __init__(self) -> None:
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0

    def __call__(
        self,
        predicted_answer: str,
        ground_truth_answers: List[str],
    ):
        exact_scores = metric_max_over_ground_truths(
            compute_exact, predicted_answer, ground_truth_answers
        )
        f1_scores = metric_max_over_ground_truths(
            compute_f1, predicted_answer, ground_truth_answers
        )

        self._total_em += int(exact_scores)
        self._total_f1 += f1_scores
        self._count += 1

    def get_metric(self, reset: bool = False) -> Tuple[float, float]:
        exact_match = self._total_em / self._count if self._count > 0 else 0
        f1_score = self._total_f1 / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return exact_match, f1_score

    def reset(self):
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0
        
        
        
class SupportMetric(Metric):
    """
    SupportMetric: Em and F1 (Similar to HotpotQA Sp metric)
    """

    def __init__(self) -> None:
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._total_precision = 0.0
        self._total_recall = 0.0
        self._count = 0

    def __call__(self, predicted_support_idxs: List[int], gold_support_idxs: List[int]):

        # Taken from hotpot_eval
        cur_sp_pred = set(map(int, predicted_support_idxs))
        gold_sp_pred = set(map(int, gold_support_idxs))
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

        # In case everything is empty, set both f1, em to be 1.0.
        # Without this change, em gets 1 and f1 gets 0
        if not cur_sp_pred and not gold_sp_pred:
            f1, em = 1.0, 1.0
            f1, em = 1.0, 1.0

        self._total_em += float(em)
        self._total_f1 += f1
        self._total_precision += prec
        self._total_recall += recall
        self._count += 1

    def get_metric(self, reset: bool = False) -> Tuple[float, float]:
        """
        Returns
        -------
        Average exact match and F1 score (in that order).
        """
        exact_match = self._total_em / self._count if self._count > 0 else 0
        f1_score = self._total_f1 / self._count if self._count > 0 else 0
        # precision_score = self._total_precision / self._count if self._count > 0 else 0
        # recall_score = self._total_recall / self._count if self._count > 0 else 0

        if reset:
            self.reset()
        return exact_match, f1_score

    def reset(self):
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._total_precision = 0.0
        self._total_recall = 0.0
        self._count = 0
        
        
# https://github.com/StonyBrookNLP/musique/blob/main/evaluate_v1.0.py#L62
def compute_metrics(prediction_instances, ground_truth_instances) -> Dict: #evaluate

    
    answer_metric = AnswerMetric()
    support_metric = SupportMetric()
    
    assert len(prediction_instances) == len(
        ground_truth_instances
    ), "The number of lines in the two files are not the same."

    for ground_truth_instance, prediction_instance in zip(
        ground_truth_instances, prediction_instances
    ):

        assert (
            ground_truth_instance["id"] == prediction_instance["id"]
        ), "The instances (ids) in prediction and gold filepath jsonl should be in same order."

        question_id = ground_truth_instance["id"]

        predicted_answer = prediction_instance["predicted_answer"]
        ground_truth_answers = [
            ground_truth_instance["answer"]
        ] + ground_truth_instance["answer_aliases"]

        predicted_support_indices = prediction_instance["predicted_support_idxs"]
        ground_truth_support_indices = ground_truth_instance["supp_para_ids"]
        #[paragraph["idx"] for paragraph in ground_truth_instance["paragraphs"] if paragraph["is_supporting"]]


        answer_metric(predicted_answer, ground_truth_answers)
        support_metric(predicted_support_indices, ground_truth_support_indices)

        

        
    metrics = {}
    metrics["answer_f1"] = answer_metric.get_metric()[1]
    metrics["support_f1"] = support_metric.get_metric()[1]
    metrics["answer_em"] = answer_metric.get_metric()[0]
    metrics["support_em"] = support_metric.get_metric()[0]

    
    return metrics



#https://github.com/allenai/allennlp/blob/80fb6061e568cb9d6ab5d45b661e86eb61b92c82/allennlp/nn/util.py#LL1547C1-L1555C55
def get_range_vector(size: int, device: int) -> torch.Tensor:
    """
    Returns a range vector with the desired size, starting at 0. The CUDA implementation
    is meant to avoid copy data from CPU to GPU.
    """
    if device > -1:
        return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
    else:
        return torch.arange(0, size, dtype=torch.long)
    
    
#https://github.com/StonyBrookNLP/musique/blob/bec107f5756f067657778ee89a9add6902191e7a/allennlp_lib/models/utils.py#LL6C1-L47C71    
def get_best_span(
        span_start_logits: torch.Tensor,
        span_end_logits: torch.Tensor,
        max_length: int = None
    ) -> torch.Tensor:
    """
    This acts the same as the static method ``BidirectionalAttentionFlow.get_best_span()``
    in ``allennlp/models/reading_comprehension/bidaf.py``. We keep it here so that users can
    directly import this function without the class.

    We call the inputs "logits" - they could either be unnormalized logits or normalized log
    probabilities.  A log_softmax operation is a constant shifting of the entire logit
    vector, so taking an argmax over either one gives the same result.
    """
    if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
        raise ValueError("Input shapes must be (batch_size, passage_length)")
    batch_size, passage_length = span_start_logits.size()
    assert batch_size == 1, 'bs > 1'
    device = span_start_logits.device
    # (batch_size, passage_length, passage_length)
    span_log_probs = span_start_logits.unsqueeze(2) + span_end_logits.unsqueeze(1)
    # Only the upper triangle of the span matrix is valid; the lower triangle has entries where
    # the span ends before it starts.
    span_mask = torch.triu(torch.ones((passage_length, passage_length), device=device))

    if max_length is not None:
        range_vector = get_range_vector(passage_length, get_device_of(span_start_logits))
        range_matrix = range_vector.unsqueeze(0)-range_vector.unsqueeze(1)
        length_mask = ((range_matrix < max_length) & (range_matrix >= 0))
        span_mask = (span_mask.long() & length_mask).float()

    span_log_mask = span_mask.log()

    valid_span_log_probs = span_log_probs + span_log_mask

    # Here we take the span matrix and flatten it, then find the best span using argmax.  We
    # can recover the start and end indices from this flattened list using simple modular
    # arithmetic.
    # (batch_size, passage_length * passage_length)
    
    #here we select best element from each sequence of the batch
    #i need the best score across batches
    best_spans = torch.max(valid_span_log_probs.view(batch_size, -1), dim=-1)#.argmax(-1)
    
    span_start_indices = best_spans.indices // passage_length
    span_end_indices = best_spans.indices % passage_length
    return torch.stack([span_start_indices, span_end_indices], dim=-1)[0].detach().cpu().numpy(), best_spans.values[0].detach().cpu().numpy().tolist()

#https://github.com/allenai/allennlp/blob/80fb6061e568cb9d6ab5d45b661e86eb61b92c82/allennlp/nn/util.py#LL1244C1-L1251C35
def get_device_of(tensor: torch.Tensor) -> int:
    """
    Returns the device of the tensor.
    """
    if not tensor.is_cuda:
        return -1
    else:
        return tensor.get_device()
    
#https://github.com/allenai/allennlp/blob/80fb6061e568cb9d6ab5d45b661e86eb61b92c82/allennlp/nn/util.py#LL856C1-L872C5
def replace_masked_values(
    tensor: torch.Tensor, mask: torch.BoolTensor, replace_with: float
) -> torch.Tensor:
    """
    Replaces all masked values in `tensor` with `replace_with`.  `mask` must be broadcastable
    to the same shape as `tensor`. We require that `tensor.dim() == mask.dim()`, as otherwise we
    won't know which dimensions of the mask to unsqueeze.

    This just does `tensor.masked_fill()`, except the pytorch method fills in things with a mask
    value of 1, where we want the opposite.  You can do this in your own code with
    `tensor.masked_fill(~mask, replace_with)`.
    """
    if tensor.dim() != mask.dim():
        raise ConfigurationError(
            "tensor.dim() (%d) != mask.dim() (%d)" % (tensor.dim(), mask.dim())
        )
    return tensor.masked_fill(~mask, replace_with)


def info_value_of_dtype(dtype: torch.dtype):
    """
    Returns the `finfo` or `iinfo` object of a given PyTorch data type. Does not allow torch.bool.
    """
    if dtype == torch.bool:
        raise TypeError("Does not support torch.bool")
    elif dtype.is_floating_point:
        return torch.finfo(dtype)
    else:
        return torch.iinfo(dtype)


#https://github.com/allenai/allennlp/blob/80fb6061e568cb9d6ab5d45b661e86eb61b92c82/allennlp/nn/util.py#L2080
def min_value_of_dtype(dtype: torch.dtype):
    """
    Returns the minimum value of a given PyTorch data type. Does not allow torch.bool.
    """
    return info_value_of_dtype(dtype).min




#https://github.com/StonyBrookNLP/musique/blob/bec107f5756f067657778ee89a9add6902191e7a/allennlp_lib/models/utils.py#LL50C1-L86C71
def get_best_k_spans(
        span_start_logits: torch.Tensor,
        span_end_logits: torch.Tensor,
        num_spans: int,
        max_length: int = None
    ) -> torch.Tensor:

    best_spans = []
    range_vector = get_range_vector(span_start_logits.shape[0], get_device_of(span_start_logits))
    for _ in range(num_spans):
        best_span = get_best_span(span_start_logits, span_end_logits, max_length)

        mask = torch.ones_like(span_start_logits, dtype=torch.bool)

        # Option 1
        mask[range_vector, best_span[:, 0]] = False
        mask[range_vector, best_span[:, 1]] = False

        # # Option 2 (seems too extreme)
        # for i, (start, end) in enumerate(best_span):
        #     mask[i, start : end + 1] = False

        span_start_logits = replace_masked_values_with_big_negative_number(span_start_logits, mask)
        span_end_logits = replace_masked_values_with_big_negative_number(span_end_logits, mask)

        best_spans.append(best_span)

    best_spans = torch.stack(best_spans, dim=1)
    return best_spans


def replace_masked_values_with_big_negative_number(x: torch.Tensor, mask: torch.Tensor):
    """
    Replace the masked values in a tensor something really negative so that they won't
    affect a max operation.
    """
    return replace_masked_values(x, mask, min_value_of_dtype(x.dtype))