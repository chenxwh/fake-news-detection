"""
Created on 03 Sep 2021
author: Chenxi
"""

from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig
from transformers import BertTokenizer, BertForSequenceClassification
from model import KnowbertForSequenceClassification, ErnieForSequenceClassification, KAdaptersForSequenceClassification
from ernie.code.knowledge_bert import BertTokenizer as ErnieBertTokenizer


def get_tokenizer_and_model(config):
    if config['model_name'] == 'roberta-base':
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=config['num_labels'])
    elif config['model_name'] == 'roberta-large':
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        model = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=config['num_labels'])
    elif config['model_name'] == 'bert-base':
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=config['num_labels'])
    elif config['model_name'] == 'bert-base-cased':
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=config['num_labels'])
    elif config['model_name'] in ['knowbert-w-w', 'knowbert-wiki', 'knowbert-wordnet']:
        model_archive = config['model_archive'][config['model_name']]
        model = KnowbertForSequenceClassification(model_archive, config)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif config['model_name'] in ['kepler', 'erica-roberta']:
        model_config_path = config['model_archive'][config['model_name']]
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        model = RobertaForSequenceClassification.from_pretrained(model_config_path, from_tf=False, num_labels=config['num_labels'])
    elif config['model_name'] in ['k_adapter']:
        # k_adapter are based on roberta-large by default
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        model = KAdaptersForSequenceClassification(config['args'], config)
    elif config['model_name'] in ['ernie']:
        model_config_path = config['model_archive'][config['model_name']]
        tokenizer = ErnieBertTokenizer.from_pretrained(model_config_path, do_lower_case=True)
        model, _ = ErnieForSequenceClassification.from_pretrained(model_config_path, num_labels=config['num_labels'])

    return tokenizer, model