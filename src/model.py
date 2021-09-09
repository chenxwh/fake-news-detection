"""
Created on 02 Sep 2021
author: Chenxi
"""

import os
import torch
import torch.nn as nn
from allennlp.data import Instance
from allennlp.data.fields import LabelField
from kb.include_all import ModelArchiveFromParams
from allennlp.common import Params
from transformers.modeling_outputs import SequenceClassifierOutput
from kb.knowbert_utils import KnowBertBatchifier
from ernie.code.knowledge_bert.modeling import PreTrainedBertModel as ErniePreTrainedBertModel
from ernie.code.knowledge_bert.modeling import BertModel
from k_adapter.pytorch_transformers import RobertaModel
from k_adapter.pytorch_transformers.my_modeling_roberta import AdapterModel


class ErnieForSequenceClassification(ErniePreTrainedBertModel):
    def __init__(self, config, num_labels=2):
        super(ErnieForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, input_ent=None, ent_mask=None, labels=None):
        if input_ent is None:
            input_ent = torch.zeros_like(input_ids)
        if ent_mask is None:
            ent_mask = torch.zeros_like(input_ids)
        _, pooled_output = self.bert(input_ids, token_type_ids=None, attention_mask=None,
                                     input_ent=input_ent, ent_mask=ent_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            # for single_label_classification
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )


class KnowbertForSequenceClassification(nn.Module):
    def __init__(self, archive_file, config):
        super(KnowbertForSequenceClassification, self).__init__()
        # load knowbert model
        params = Params({"archive_file": archive_file})
        self.knowbert = ModelArchiveFromParams.from_params(params=params)
        self.num_labels = config['num_labels']
        self.dropout = nn.Dropout(config['dropout_prob'])
        self.classifier = nn.Linear(config['hidden_size'], config['num_labels'])

    def forward(self, tokens, segment_ids, candidates, label_ids):
        model_output = self.knowbert(tokens=tokens, segment_ids=segment_ids, candidates=candidates)
        pooled_output = model_output['pooled_output']
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if label_ids is not None:
            # for single_label_classification
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )


class CustomKnowBertBatchifier(KnowBertBatchifier):
    def iter_batches(self, statements, labels):
        # create instances
        instances = []
        for statement, label in zip(statements, labels):
            tokens_candidates = self.tokenizer_and_candidate_generator. \
                tokenize_and_generate_candidates(self._replace_mask(statement))

            fields = self.tokenizer_and_candidate_generator. \
                convert_tokens_candidates_to_fields(tokens_candidates)
            fields['label_ids'] = LabelField(label, skip_indexing=True)
            instances.append(Instance(fields))
        for batch in self.iterator(instances, num_epochs=1, shuffle=False):
            yield batch


class KAdaptersForSequenceClassification(nn.Module):
    def __init__(self, k_adapter_args, config):
        super(KAdaptersForSequenceClassification, self).__init__()
        # load knowbert model
        self.k_adapter = RobertaModelwithAdapter(k_adapter_args)
        self.num_labels = config['num_labels']
        self.dropout = nn.Dropout(config['dropout_prob'])
        self.classifier = nn.Linear(config['hidden_size_large'], config['num_labels'])

    def forward(self, input_ids=None, attention_mask=None):
        # the output is the final layer with all the hidden tensors, size(batch, max_len, hidden_size) = (4, 64, 1024)
        # now need to pool the first position out, get (4, 1024), so need to pull the second dimension's first position
        output = self.k_adapter(input_ids=input_ids, attention_mask=attention_mask)
        # k_adapter return a tuple of size 1, get the first tensor out which is size of (4, 64, 1024)
        output = output[0]
        pooled_output = [output[i][0] for i in range(len(output))]
        pooled_output = torch.stack(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class RobertaModelwithAdapter(nn.Module):
    base_model_prefix = "roberta"

    def __init__(self, args):
        super(RobertaModelwithAdapter, self).__init__()
        if not os.path.exists('roberta-large-model.pth'):
            self.model = RobertaModel.from_pretrained("roberta-large", force_download=True, output_hidden_states=True)
            torch.save(self.model, 'roberta-large-model.pth')
        self.model = torch.load('roberta-large-model.pth')
        self.config = self.model.config
        self.config.freeze_adapter = args.freeze_adapter
        self.args = args
        if args.freeze_bert:
            for p in self.model.parameters():
                p.requires_grad = False

        if args.meta_fac_adaptermodel:
            fac_adapter = AdapterModel(self.args, self.config)
            fac_adapter = self.load_pretrained_adapter(fac_adapter, self.args.meta_fac_adaptermodel)
        else:
            fac_adapter = None

        if args.meta_lin_adaptermodel:
            lin_adapter = AdapterModel(self.args, self.config)
            lin_adapter = self.load_pretrained_adapter(lin_adapter, self.args.meta_lin_adaptermodel)
        else:
            lin_adapter = None

        self.fac_adapter = fac_adapter
        self.lin_adapter = lin_adapter
        if args.freeze_adapter and (self.fac_adapter is not None):
            for p in self.fac_adapter.parameters():
                p.requires_grad = False
        if args.freeze_adapter and (self.lin_adapter is not None):
            for p in self.lin_adapter.parameters():
                p.requires_grad = False

        self.adapter_num = 0
        if self.fac_adapter is not None:
            self.adapter_num += 1
        if self.lin_adapter is not None:
            self.adapter_num += 1

        if self.args.fusion_mode == 'concat':
            if args.meta_fac_adaptermodel:
                self.task_dense_fac = nn.Linear(self.config.hidden_size + self.config.hidden_size,
                                                self.config.hidden_size)
            if args.meta_lin_adaptermodel:
                self.task_dense_lin = nn.Linear(self.config.hidden_size + self.config.hidden_size,
                                                self.config.hidden_size)
            self.task_dense = nn.Linear(self.config.hidden_size + self.config.hidden_size, self.config.hidden_size)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids,
                             attention_mask=attention_mask)

        pretrained_model_last_hidden_states = outputs[0]  # original roberta output

        if self.fac_adapter is not None:
            fac_adapter_outputs, _ = self.fac_adapter(outputs)
        if self.lin_adapter is not None:
            lin_adapter_outputs, _ = self.lin_adapter(outputs)

        if self.args.fusion_mode == 'add':
            task_features = pretrained_model_last_hidden_states
            if self.fac_adapter is not None:
                task_features = task_features + fac_adapter_outputs
            if self.lin_adapter is not None:
                task_features = task_features + lin_adapter_outputs
        elif self.args.fusion_mode == 'concat':
            combine_features = pretrained_model_last_hidden_states
            if self.args.meta_fac_adaptermodel:
                fac_features = self.task_dense_fac(torch.cat([combine_features, fac_adapter_outputs], dim=2))
                task_features = fac_features
            if self.args.meta_lin_adaptermodel:
                lin_features = self.task_dense_lin(torch.cat([combine_features, lin_adapter_outputs], dim=2))
                task_features = lin_features
            if (self.fac_adapter is not None) and (self.lin_adapter is not None):
                task_features = self.task_dense(torch.cat([fac_features, lin_features], dim=2))
        return (task_features,)

    def load_pretrained_adapter(self, adapter, adapter_path):
        new_adapter = adapter
        model_dict = new_adapter.state_dict()
        adapter_meta_dict = torch.load(adapter_path, map_location=lambda storage, loc: storage)

        for item in ['out_proj.bias', 'out_proj.weight', 'dense.weight', 'dense.bias']:
            if item in adapter_meta_dict:
                adapter_meta_dict.pop(item)

        changed_adapter_meta = {}
        for key in adapter_meta_dict.keys():
            changed_adapter_meta[key.replace('adapter.', 'adapter.')] = adapter_meta_dict[key]
            # changed_adapter_meta[key.replace('model.','roberta.')] = adapter_meta_dict[key]
        changed_adapter_meta = {k: v for k, v in changed_adapter_meta.items() if k in model_dict.keys()}
        model_dict.update(changed_adapter_meta)
        new_adapter.load_state_dict(model_dict)
        return new_adapter
