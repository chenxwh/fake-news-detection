"""
Created on 02 Sep 2021
author: Chenxi
"""

import os
import argparse
import torch
import random
import numpy as np
import pandas as pd
import yaml
import yamlordereddictloader
from datetime import datetime
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import classification_report
from dataset import NewsDataset
from trainer import Trainer
from predictor import Predictor


def get_tokernizer_and_model(config):
    if config['model_name'] == 'roberta-base':
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        model = RobertaForSequenceClassification.from_pretrained('roberta-base',
                                                                 num_labels=config[config['dataset_name']]['num_classes'])
    elif config['model_name'] == 'bert-base':
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        model = BertForSequenceClassification.from_pretrained('bert-base-cased',
                                                              num_labels=config[config['dataset_name']]['num_classes'])
    return tokenizer, model


def save_report(config, report_dict):
    report_path = os.path.join(config['report_path'], config['dataset_name'], config['model_name'])
    os.makedirs(report_path, exist_ok=True)
    report_df = pd.DataFrame(report_dict).transpose()
    now = datetime.now()
    date_time = now.strftime("%Y%m%d%H%M%S")
    report_file_name = os.path.join(report_path, date_time + '.txt')
    report_df.to_csv(report_file_name)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    np.random.seed(seed)              # Numpy module
    random.seed(seed)                 # Python random module
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='covid', choices=['liar', 'covid'])
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--mode', type=str, choices=['train', 'test'])
    parser.add_argument('--model', type=str, default='roberta-base', choices=['roberta-base', 'bert-base'])
    parser.add_argument('--verbose', action='store_true', default=False, help='print detailed training process')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yamlordereddictloader.Loader)

    config = dict(config)
    config['model_name'] = args.model
    config['dataset_name'] = args.dataset
    config['verbose'] = args.verbose
    set_seed(config['seed'])

    device = torch.device("cuda:{}".format(config['gpu']) if config['gpu'] != -1 else "cpu")
    config['device'] = device

    tokenizer, model = get_tokernizer_and_model(config)
    model.to(device)

    if args.mode == 'train':
        train_data = NewsDataset(config, args.dataset, tokenizer, 'train')
        val_data = NewsDataset(config, args.dataset, tokenizer, 'val')
        train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=config['batch_size'])
        val_dataloader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=config['batch_size'])

        trainer = Trainer(config, model, train_dataloader, val_dataloader)
        trainer.train(verbose=config['verbose'])
    else:
        test_data = NewsDataset(config, args.dataset, tokenizer, 'test')
        test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=config['batch_size'])
        predictor = Predictor(config, model, test_dataloader)

        predictions, true_labels = predictor.predict()
        report = classification_report(true_labels, predictions, target_names=config[args.dataset]['class_names'], digits=4)
        print(report)
        report_dict = classification_report(true_labels, predictions, target_names=config[args.dataset]['class_names'],
                                            digits=4, output_dict = True)
        save_report(config, report_dict)


if __name__ == "__main__":
    main()
