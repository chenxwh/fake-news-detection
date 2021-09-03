"""
Created on 02 Sep 2021
author: Chenxi
"""

from torch.utils.data import Dataset
import torch
import pandas as pd


class NewsDataset(Dataset):
    def __init__(self, config, dataset_name, tokenizer, mode, batcher=None, load_size=None):
        self.config = config
        self.dataset_name = dataset_name
        self.labels, self.statements = load_dataset(config, dataset_name, mode, load_size)
        self.tokenizer = tokenizer
        self.bathcer = batcher

    def __len__(self):
        return len(self.statements)

    def __getitem__(self, idx):
        statement = str(self.statements[idx])
        label = self.labels[idx]

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

        if self.config['model_name'] in ['knowbert-w-w', 'knowbert-wiki', 'knowbert-wordnet']:
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long),
                'candidates': get_knowbert_candidates(self.bathcer, statement)
            }

        # up to now, all the bert, roberta and k-adapters use the standard Dataset
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def get_knowbert_candidates(batcher, statement):
    tokens_candidates = batcher.tokenizer_and_candidate_generator.tokenize_and_generate_candidates(batcher._replace_mask(statement))
    fields = batcher.tokenizer_and_candidate_generator.convert_tokens_candidates_to_fields(tokens_candidates)


def load_dataset(config, dataset_name, mode, load_size=None):
    data_df = pd.read_csv(config[dataset_name][mode], sep="\t", header=None)
    if load_size is not None:
        data_df = data_df[:load_size + 1]

    # Fill nan (empty boxes) with -1
    data_df = data_df.fillna(-1)
    data_df = data_df.to_numpy()
    statements = [data_df[i][config[dataset_name]['statement_col']] for i in range(len(data_df))]
    labels = [data_df[i][config[dataset_name]['label_col']] for i in range(len(data_df))]
    labels = convert_labels(dataset_name, labels)

    return labels, statements


def convert_labels(dataset_name, labels):
    """
    convert labels to numbers
    """
    encoded_labels = [0] * len(labels)
    if dataset_name == 'covid':
        for i in range(len(labels)):
            if labels[i] == 'real':
                encoded_labels[i] = 0
            elif labels[i] == 'fake':
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
