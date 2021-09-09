"""
Created on 02 Sep 2021
author: Chenxi
"""

from torch.utils.data import Dataset
import torch
import pandas as pd
from ast import literal_eval


class NewsDataset(Dataset):
    def __init__(self, config, tokenizer, mode, load_size=None):
        self.config = config
        self.dataset_name = config['dataset_name']
        self.entity_total = 0

        if self.config['model_name'] == 'ernie':
            self.labels, self.statements, self.ents_list = self.load_ernie_dataset(config, mode, load_size)
        else:
            self.labels, self.statements = load_dataset(config, mode, load_size)

        self.tokenizer = tokenizer


    def __len__(self):
        return len(self.statements)

    def __getitem__(self, idx):
        statement = str(self.statements[idx])
        label = self.labels[idx]

        if self.config['model_name'] == 'ernie':
            ents = self.ents_list[idx]

            return get_ernie_encoding(self.config, self.tokenizer, statement, label, ents)

        encoding = self.tokenizer.encode_plus(
            statement,
            add_special_tokens=True,
            truncation=True,
            max_length=self.config['embedding_dim'],
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def load_ernie_dataset(self, config, mode, load_size):
        data_path = config['ernie_pre_calculated'][config['dataset_name']][mode]
        data_df = pd.read_csv(data_path)
        if load_size is not None:
            data_df = data_df[:load_size]

        # Fill nan (empty boxes) with -1
        data_df = data_df.fillna(-1)
        data_df = data_df.to_numpy()
        statements = [data_df[i][config[config['dataset_name']]['statement_col'] + 1] for i in range(len(data_df))]
        labels = [data_df[i][config[config['dataset_name']]['label_col'] + 1] for i in range(len(data_df))]
        # here filter the ents only above the threshold, ents[-1] is the score
        valid_ents = []
        for i in range(len(data_df)):
            all_ents_i = literal_eval(data_df[i][-1])
            valid_ents_i = [ent for ent in all_ents_i if ent[-1] > config['tagme_thr']]
            valid_ents.append(valid_ents_i)
            self.entity_total += len(valid_ents_i)

        labels = convert_labels(config, labels)

        return labels, statements, valid_ents


def get_ernie_encoding(config, tokenizer, statement, label, statement_entities):
    max_seq_length = config['embedding_dim']

    tokens_statement, entities = tokenizer.tokenize(statement, statement_entities)
    if len(tokens_statement) > max_seq_length - 2:
        tokens_statement = tokens_statement[:(max_seq_length - 2)]
        entities = entities[:(max_seq_length - 2)]

    tokens = ["[CLS]"] + tokens_statement + ["[SEP]"]
    ents = ["UNK"] + entities + ["UNK"]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ent = []
    ent_mask = []

    entity2id = config['entity2id']


    for ent in ents:
        if ent != "UNK" and ent in entity2id:
            input_ent.append(entity2id[ent])
            ent_mask.append(1)
        else:
            input_ent.append(-1)
            ent_mask.append(0)
    ent_mask[0] = 1
    input_mask = [1] * len(input_ids)

    padding = [0] * (max_seq_length - len(input_ids))
    padding_ = [-1] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    input_ent += padding_
    ent_mask += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(input_ent) == max_seq_length
    assert len(ent_mask) == max_seq_length

    # Convert inputs to PyTorch tensors
    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'input_ent': torch.tensor(input_ent, dtype=torch.long),
        'ent_mask': torch.tensor(ent_mask, dtype=torch.long),
        'labels': torch.tensor(label, dtype=torch.long)
    }


def load_dataset(config, mode, load_size=None):
    data_df = pd.read_csv(config[config['dataset_name']][mode], sep="\t", header=None)
    if load_size is not None:
        data_df = data_df[:load_size]

    # Fill nan (empty boxes) with -1
    data_df = data_df.fillna(-1)
    data_df = data_df.to_numpy()
    statements = [data_df[i][config[config['dataset_name']]['statement_col']] for i in range(len(data_df))]
    labels = [data_df[i][config[config['dataset_name']]['label_col']] for i in range(len(data_df))]
    labels = convert_labels(config, labels)

    return labels, statements





def convert_labels(config, labels):
    """
    convert labels to numbers
    """
    encoded_labels = [0] * len(labels)
    if config['dataset_name'] == 'covid':
        for i in range(len(labels)):
            if labels[i] == 'real':
                encoded_labels[i] = 0
            elif labels[i] == 'fake':
                encoded_labels[i] = 1
    else:
        if config['num_labels'] == 2:
            for i in range(len(labels)):
                if labels[i] in ['true', 'mostly-true', 'half-true']:
                    encoded_labels[i] = 0
                elif labels[i] in ['barely-true', 'false', 'pants-fire']:
                    encoded_labels[i] = 1
        else:
            for i in range(len(labels)):
                if labels[i] == 'true':
                    encoded_labels[i] = 0
                elif labels[i] == 'mostly-true':
                    encoded_labels[i] = 1
                elif labels[i] == 'half-true':
                    encoded_labels[i] = 2
                elif labels[i] == 'barely-true':
                    encoded_labels[i] = 3
                elif labels[i] == 'false':
                    encoded_labels[i] = 4
                elif labels[i] == 'pants-fire':
                    encoded_labels[i] = 5
    return encoded_labels
