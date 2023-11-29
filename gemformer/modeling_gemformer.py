import torch
from torch import nn
from transformers import RobertaPreTrainedModel, RobertaModel


class RobertaLMHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x_emb = self.dense(features)
        x = gelu(x_emb)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x

    def _tie_weights(self):
        self.bias = self.decoder.bias


class RobertaEvidenceDetectionHead(nn.Module):

    def __init__(self, config, num_clf_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, num_clf_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dense(x)
        x = gelu(x)
        x = self.out_proj(x)
        return x


class RobertaQuestionClassificationHead(nn.Module):

    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


def or_softmax_cross_entropy_loss_one_doc(logits, target, ignore_index=-1, dim=-1, reduction='sum'):
        """
        from the Longformer code: 
        loss function suggested in section 2.2 here https://arxiv.org/pdf/1710.10723.pdf
        """
        assert logits.ndim == 2, f"{logits.shape}, {logits}"
        assert target.ndim == 2
        assert logits.size(0) == target.size(0)

        # with regular CrossEntropyLoss, the numerator is only one of the logits specified by the target
        # here, the numerator is the sum of a few potential targets, where some of them is the correct answer

        # compute a target mask
        target_mask = target == ignore_index
        # replaces ignore_index with 0, so `gather` will select logit at index 0 for the msked targets
        masked_target = target * (1 - target_mask.long())
        # gather logits
        gathered_logits = logits.gather(dim=dim, index=masked_target)
        # Apply the mask to gathered_logits. Use a mask of -inf because exp(-inf) = 0
        gathered_logits[target_mask] = float('-inf')
        
        # each batch is one example
        gathered_logits = gathered_logits.view(1, -1)
        logits = logits.view(1, -1)
        
        # numerator = log(sum(exp(gathered logits)))
        log_score = torch.logsumexp(gathered_logits, dim=dim, keepdim=False)
        # denominator = log(sum(exp(logits)))
        log_norm = torch.logsumexp(logits, dim=dim, keepdim=False)

        # compute the loss
        loss = -(log_score - log_norm)

        # some of the examples might have a loss of `inf` when `target` is all `ignore_index`.
        # remove those from the loss before computing the sum. Use sum instead of mean because
        # it is easier to compute
        if reduction == 'sum':
            return loss[~torch.isinf(loss)].sum()
        elif reduction == 'none':
            return loss[~torch.isinf(loss)]
        else:
            print('you must specify the reduction: either sum or none')
            return None


class RobertaGEMFormer(RobertaPreTrainedModel):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]

    def __init__(self, config, paragraph_marker_token, sentence_marker_token, question_type_num_labels,
                 ques_type_loss_weight=1., ans_loss_weight=1.,
                 para_loss_weight=1., sent_loss_weight=1., supp_bce_loss=None, ans_ce_loss=False,
                ):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.paragraph_marker_token = paragraph_marker_token
        self.sentence_marker_token = sentence_marker_token
        self.question_type_num_labels = question_type_num_labels
        self.ques_type_loss_weight = ques_type_loss_weight
        self.ans_loss_weight = ans_loss_weight
        self.para_loss_weight = para_loss_weight
        self.sent_loss_weight = sent_loss_weight
        self.supp_bce_loss = supp_bce_loss
        self.ans_ce_loss = ans_ce_loss

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)
        
        if self.question_type_num_labels is not None:
            self.question_classifier = RobertaQuestionClassificationHead(
                config,num_labels=question_type_num_labels
            )
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        if self.supp_bce_loss == True:
            self.paragraph_classifier = RobertaEvidenceDetectionHead(config, 1)
        else:
            self.paragraph_classifier = RobertaEvidenceDetectionHead(config, 2)

        if self.sentence_marker_token is not None:
            if self.supp_bce_loss == True:
                self.sentence_classifier = RobertaEvidenceDetectionHead(config, 1)
            else:
                self.sentence_classifier = RobertaEvidenceDetectionHead(config, 2)

        for param in self.lm_head.parameters():
            param.requires_grad = False
    
        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        context_start_id: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        question_type_labels: Optional[torch.LongTensor] = None,
        paragraph_labels: Optional[torch.LongTensor] = None,
        sentence_labels: Optional[torch.LongTensor] = None,
        mem_tokens: Optional[torch.LongTensor] = None,
        eval_flag: Optional[bool] = False,
        return_lm_logits: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        minimize_pad: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], QuestClsQASpanSuppClsFullDocModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        question_type_logits = None
        start_logits = None
        end_logits = None
        paragraph_logits = None
        sentence_logits = None
        lm_logits = None
        bs, seq_len = input_ids.shape

        if not (return_lm_logits or output_hidden_states):
            if paragraph_labels is not None:
                para_len = paragraph_labels.shape[1]
            else:
                para_len = 0

            if sentence_labels is not None:
                sent_len = sentence_labels.shape[1]
            else:
                sent_len = 0

        if context_start_id is not None:
            mem_start_id = context_start_id[0]

        if mem_tokens is not None:
            assert len(mem_tokens.shape) == 1, f"mem_tokens shape = {mem_tokens.shape}"
            mem_len = mem_tokens.shape[-1]
            if mem_len > 0:
                mem_tokens = mem_tokens.repeat(bs, 1).to(mem_tokens.device)
                mem_tokens_mask = torch.ones_like(mem_tokens).to(attention_mask.device)

                input_ids = torch.cat(
                    (input_ids[:, :mem_start_id], mem_tokens,
                     input_ids[:, mem_start_id:]), -1)
                attention_mask = torch.cat([mem_tokens_mask, attention_mask], -1)
        else:
            mem_len = 0

        #get paragraph tokens positions from inputs to calculate loss only for paragraph marker tokens
        if not (return_lm_logits or output_hidden_states):
            paragraph_positions = []
            sentence_positions = []

            if paragraph_labels is not None:
                paragraph_positions = [torch.where(i == self.paragraph_marker_token)[0] for i in input_ids]

            if sentence_labels is not None:
                sentence_positions = [torch.where(i == self.sentence_marker_token)[0] for i in input_ids]

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs[0]

        if return_lm_logits:
            lm_logits = self.lm_head(sequence_output)
            if not return_dict:
                output = (lm_logits,) + outputs[2:]
                return ((lm_logits,) + output) if total_loss is not None else output

            return QuestClsQASpanSuppClsFullDocModelOutput(lm_logits=lm_logits)

        elif output_hidden_states:
            if not return_dict:
                output = outputs[2:]
                return output

            return QuestClsQASpanSuppClsFullDocModelOutput(
                hidden_states=outputs.hidden_states
            )
        else:

            question_type_logits = None
            if self.question_type_num_labels is not None:
                question_type_logits = self.question_classifier(sequence_output)
                question_type_logits = question_type_logits.sum(0) / bs

            answer_span_logits = self.qa_outputs(sequence_output)
            start_logits, end_logits = answer_span_logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1).contiguous()
            end_logits = end_logits.squeeze(-1).contiguous()
            # drop memory positions before loss calculation
            start_logits = torch.cat([start_logits[:, :mem_start_id],
                                      start_logits[:, mem_start_id + mem_len:]
                                     ], -1)
            end_logits = torch.cat([end_logits[:, :mem_start_id],
                                    end_logits[:, mem_start_id + mem_len:]
                                   ], -1)

            # supporting evidence titles logits
            if paragraph_labels is not None:
                paragraph_logits = self.paragraph_classifier(sequence_output)

                if self.supp_bce_loss == True:
                    # labels for bce loss
                    bce_para_labels = []
                    for i in paragraph_labels:
                        bce_para_labels += [label for label in i if label != -1]
                    bce_para_labels = torch.stack(bce_para_labels).to(paragraph_labels.device,
                                                                      dtype=torch.float32)
                    paragraph_labels = bce_para_labels
                    para_logits_list = []
                    for logits, pos in zip(paragraph_logits, paragraph_positions):
                        para_logits_list += [logits[pp] for pp in pos]
                    paragraph_logits = torch.stack(para_logits_list).to(paragraph_logits.device,
                                                                        dtype=paragraph_logits.dtype)
                else:
                    try:
                        paragraph_logits = torch.stack([
                    torch.stack([logits[pp] for pp in pos] + \
                                [torch.ones(logits.shape[-1]).to(logits.device) * float('-inf')] * \
                                (para_len - len(pos))).to(dtype=logits.dtype) for logits, pos in zip(paragraph_logits,
                                paragraph_positions)
                                ]).to(paragraph_labels.device)
                    except:
                        print(f"paragraph_positions = {paragraph_positions}",
                              'paragraph logits stack failed', [[logits[pp] for pp in pos] + \
                              [torch.ones(logits.shape[-1]).to(logits.device) * float('-inf')] * \
                              (para_len - len(pos)) for logits, pos in zip(paragraph_logits,
                                                                           paragraph_positions)
                              ])

            # supporting evidence sentences logits
            if sentence_labels is not None:
                sentence_logits = self.sentence_classifier(sequence_output)
                if self.supp_bce_loss == True:
                    # labels for bce loss
                    bce_sent_labels = []
                    for i in sentence_labels:
                        bce_sent_labels += [label for label in i if label != -1]
                    bce_sent_labels = torch.stack(bce_sent_labels).to(sentence_labels.device, dtype=torch.float32)
                    sentence_labels = bce_sent_labels
                    sent_logits_list = []
                    for logits, pos in zip(sentence_logits, sentence_positions):
                        sent_logits_list += [logits[pp] for pp in pos]
                    sentence_logits = torch.stack(sent_logits_list).to(sentence_logits.device, dtype=sentence_logits.dtype)
                else:
                    sentence_logits = torch.stack([
                    torch.stack([logits[pp] for pp in pos] + \
                                [torch.ones(logits.shape[-1]).to(logits.device) * float('-inf')] * \
                                (sent_len - len(pos))).to(dtype=logits.dtype) for logits, pos in zip(sentence_logits,
                                sentence_positions)
                                ]).to(sentence_labels.device)

            total_loss = None

            answer_span_loss = None
            if start_positions is not None and end_positions is not None  and (not eval_flag):
                # If we are on multi-GPU, split add a dimension
                if len(start_positions.size()) > 1:
                    start_positions = start_positions.squeeze(-1)
                if len(end_positions.size()) > 1:
                    end_positions = end_positions.squeeze(-1)
                if self.ans_ce_loss == True:
                    ans_loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='sum')
                    start_loss = ans_loss_fct(start_logits, start_positions)
                    end_loss = ans_loss_fct(end_logits, end_positions)
                else:
                    # loss function suggested in section 2.2 here https://arxiv.org/pdf/1710.10723.pdf
                    # NOTE: this returns sum of losses, not mean, so loss won't be normalized across different batch sizes
                    # but batch size is always 1, so this is not a problem
                    start_loss = or_softmax_cross_entropy_loss_one_doc(start_logits, start_positions, ignore_index=-1)
                    end_loss = or_softmax_cross_entropy_loss_one_doc(end_logits, end_positions, ignore_index=-1)
                answer_span_loss = (start_loss + end_loss) / 2
                total_loss = answer_span_loss * self.ans_loss_weight

            question_type_loss = None
            if self.question_type_num_labels is not None:
                if ((question_type_labels is not None) and (not eval_flag)):
                    question_type_loss_fct = CrossEntropyLoss(reduction='sum')
                    question_type_loss = question_type_loss_fct(question_type_logits.view(-1, self.question_type_num_labels),
                                                                question_type_labels.view(-1))
                    total_loss = total_loss + (question_type_loss * self.ques_type_loss_weight)

            paragraph_loss = None
            if paragraph_labels is not None and (not eval_flag):
                if self.supp_bce_loss == True:
                    paragraph_logits = paragraph_logits.squeeze(-1)
                    paragraph_loss_fct = BCEWithLogitsLoss(reduction='sum')
                    paragraph_loss = paragraph_loss_fct(paragraph_logits, paragraph_labels)
                else:
                    paragraph_loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='sum')
                    paragraph_loss = paragraph_loss_fct(paragraph_logits.view(-1, 2), paragraph_labels.view(-1))

                total_loss = total_loss + (paragraph_loss * self.para_loss_weight)

            sentence_loss = None
            if sentence_labels is not None and (not eval_flag):
                if self.supp_bce_loss == True:
                    sentence_logits = sentence_logits.squeeze(-1)
                    sentence_loss_fct = BCEWithLogitsLoss(reduction='sum')
                    sentence_loss = sentence_loss_fct(sentence_logits, sentence_labels)
                else:
                    sentence_loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='sum')
                    sentence_loss = sentence_loss_fct(sentence_logits.view(-1, 2), sentence_labels.view(-1))

                total_loss = total_loss + (sentence_loss * self.sent_loss_weight)

            if not return_dict:
                output = (total_loss,) + outputs[2:]
                return ((total_loss,) + output) if total_loss is not None else output

            return QuestClsQASpanSuppClsFullDocModelOutput(
                loss=total_loss,
                question_type_loss=question_type_loss,
                answer_span_loss=answer_span_loss,
                supp_paragraph_loss=paragraph_loss,
                supp_sentence_loss=sentence_loss,
                question_type_logits=question_type_logits.unsqueeze(0) if question_type_logits is not None else None,
                start_logits=start_logits,
                end_logits=end_logits,
                supp_paragraph_logits=paragraph_logits,
                supp_sentence_logits=sentence_logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)
        if past is not None:
            input_ids = input_ids[:, -1:]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past}

    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past
