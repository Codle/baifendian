from typing import Optional, Union

import torch
import torch.nn as nn

import texar.torch as tx


class BertClassifier(nn.Module):

    def __init__(self, cfg):
        super(BertClassifier, self).__init__()

        self.num_labels = cfg['num_labels']
        self.bert_encoder = tx.modules.BERTEncoder('bert-base-chinese')
        self.dropout = nn.Dropout(cfg['hidden_dropout_prob'])
        self.classifier = nn.Linear(cfg['hidden_size'],
                                    cfg['num_labels'])

    def forward(self,
                inputs: Union[torch.Tensor, torch.LongTensor],
                labels: Optional[torch.LongTensor] = None,
                sequence_length: Optional[torch.LongTensor] = None,
                segment_ids: Optional[torch.LongTensor] = None):

        outputs = self.bert_encoder(inputs=inputs,
                                    sequence_length=sequence_length,
                                    segment_ids=segment_ids)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1).long())
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
