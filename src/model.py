"""
Created on 02 Sep 2021
author: Chenxi
"""

import os
import torch
import torch.nn as nn
from kb.include_all import ModelArchiveFromParams
from allennlp.common import Params
from transformers.modeling_outputs import SequenceClassifierOutput
from k_adapters.pytorch_transformers import RobertaModel
from k_adapters.pytorch_transformers.my_modeling_roberta import AdapterModel


class KnowbertForSequenceClassification(nn.Module):
    def __init__(self, archive_file, opts):
        super(KnowbertForSequenceClassification, self).__init__()
        # load knowbert model
        params = Params({"archive_file": archive_file})
        self.knowbert = ModelArchiveFromParams.from_params(params=params)
        self.num_labels = opts['num_labels']
        self.dropout = nn.Dropout(opts['dropout_prob'])
        self.classifier = nn.Linear(opts['hidden_size'], opts['num_labels'])

    def forward(self, input_ids=None, candidates=None, labels=None):
        outputs = self.knowbert(tokens=input_ids, candidates=candidates)
        pooled_output = outputs['pooled_output']
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


class KAdaptersForSequenceClassification(nn.Module):
    def __init__(self, k_adapters_args, opts):
        super(KAdaptersForSequenceClassification, self).__init__()
        # load knowbert model
        self.k_adapters = RobertaModelwithAdapter(k_adapters_args)
        self.num_labels = opts['num_labels']
        self.dropout = nn.Dropout(opts['dropout_prob'])
        self.classifier = nn.Linear(opts['hidden_size'], opts['num_labels'])

    def forward(self, input_ids=None, attention_mask=None):
        # the output is the final layer with all the hidden tensors, size(batch, max_len, hidden_size) = (4, 64, 1024)
        # now need to pool the first position out, get (4, 1024), so need to pull the second dimension's first position
        output = self.k_adapters(input_ids=input_ids, attention_mask=attention_mask)
        # k_adapters return a tuple of size 1, get the first tensor out which is size of (4, 64, 1024)
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
        # if input_ids[:, 0].sum().item() != 0:
        #     logger.warning("A sequence with no special tokens has been passed to the RoBERTa model. "
        #                    "This model requires special tokens in order to work. "
        #                    "Please specify add_special_tokens=True in your encoding.")
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

        for item in ['out_proj.bias', 'out_proj.weight', 'dense.weight',
                     'dense.bias']:  # 'adapter.down_project.weight','adapter.down_project.bias','adapter.up_project.weight','adapter.up_project.bias'
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
    #
    # def save_pretrained(self, save_directory):
    #     """ Save a model and its configuration file to a directory, so that it
    #     """
    #     assert os.path.isdir(save_directory), "Saving path should be a directory where the model and configuration can be saved"
    #
    #     # Only save the model it-self if we are using distributed training
    #     model_to_save = self.module if hasattr(self, 'module') else self
    #
    #     # Save configuration file
    #     model_to_save.config.save_pretrained(save_directory)
    #
    #     # If we save using the predefined names, we can load using `from_pretrained`
    #     output_model_file = os.path.join(save_directory, "pytorch_model.bin")
    #
    #     torch.save(model_to_save.state_dict(), output_model_file)
    #     logger.info("Saving model checkpoint to %s", save_directory)
