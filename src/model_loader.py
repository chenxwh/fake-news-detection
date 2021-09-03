"""
Created on 03 Sep 2021
author: Chenxi
"""


from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from model import KnowbertForSequenceClassification, KAdaptersForSequenceClassification



def get_tokenizer_and_model(config):
    opts = {'num_labels': config[config['dataset_name']]['num_classes'],
            'hidden_size': config['hidden_size'],
            'dropout_prob': config['dropout_prob']}
    if config['model_name'] == 'roberta-base':
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=opts['num_labels'])
    if config['model_name'] == 'roberta-large':
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        model = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=opts['num_labels'])
    elif config['model_name'] == 'bert-base':
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=opts['num_labels'])
    elif config['model_name'] == 'bert-base-uncased':
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=opts['num_labels'])
    elif config['model_name'] in ['knowbert-w-w', 'knowbert-wiki', 'knowbert-wordnet']:
        model_archive = config['model_archive']['model_name']
        model = KnowbertForSequenceClassification(model_archive, opts)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif config['model_name'] in ['k_adapters']:
        # k_adapters are based on roberta-large by default
        # changed force_download=True in line 849 of k_adapters.pytorch_transformers.my_modeling_roberta so no
        # complaints about not enough space
        opts['hidden_size'] = config['hidden_size_large']
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        model = KAdaptersForSequenceClassification(config['args'], opts)

    return tokenizer, model