"""
Created on 02 Sep 2021
author: Chenxi
"""

import os
import sys

src_dir = os.path.dirname(os.path.realpath(__file__))
project_dir = os.path.dirname(src_dir)
sys.path.append(project_dir)

import argparse
import yaml
import yamlordereddictloader
from datetime import datetime
import datetime
import pickle
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import classification_report
from model import CustomKnowBertBatchifier
from dataset import NewsDataset, load_dataset
from trainer import Trainer
from knowbert_trainer import KnowbertTrainer
from predictor import Predictor, KnowbertPredictor
from model_loader import get_tokenizer_and_model
from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='covid', choices=['liar', 'covid'])
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--num_labels', type=int, default=None)
    parser.add_argument('--model', type=str, default='roberta-base')
    parser.add_argument('--logging', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--verbose', action='store_true', default=False, help='print detailed training process')

    # ernie
    parser.add_argument('--threshold', type=float, default=0.3, help='threshold for tagme entity linking')

    # k_adapter
    parser.add_argument("--output_dir", default='./proc/roberta_adapter', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--freeze_bert", default=False, type=bool,
                        help="freeze the parameters of original model.")
    parser.add_argument("--freeze_adapter", default=True, type=bool,
                        help="freeze the parameters of adapter.")
    parser.add_argument('--fusion_mode', type=str, default='concat',
                        help='the fusion mode for bert feature and adapter feature |add|concat')
    parser.add_argument("--adapter_transformer_layers", default=2, type=int,
                        help="The transformer layers of adapter.")
    parser.add_argument("--adapter_size", default=768, type=int,
                        help="The hidden size of adapter.")
    parser.add_argument("--adapter_list", default="0,11,23", type=str,
                        help="The layer where add an adapter")
    parser.add_argument("--adapter_skip_layers", default=0, type=int,
                        help="The skip_layers of adapter according to bert layers")
    parser.add_argument("--no_cuda", default=False, action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--meta_fac_adaptermodel', default="../pretrained_models/fac-adapter/pytorch_model.bin",
                        type=str, help='the pretrained factual adapter model')
    parser.add_argument('--meta_lin_adaptermodel', default="../pretrained_models/lin-adapter/pytorch_model.bin",
                        type=str, help='the pretrained linguistic adapter model')

    args = parser.parse_args()
    return args


def main():
    torch.cuda.empty_cache()
    args = parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yamlordereddictloader.Loader)

    config = dict(config)
    device = torch.device("cuda:{}".format(config['gpu']) if config['gpu'] != -1 else "cpu")
    config['device'] = device
    config['model_name'] = args.model
    config['dataset_name'] = args.dataset
    config['verbose'] = args.verbose

    if args.epochs is not None:
        config['epochs'] = args.epochs

    if args.num_labels is not None:
        config['num_labels'] = args.num_labels
    else:
        config['num_labels'] = config[args.dataset]['num_labels']

    config['logging_file'] = ''
    if args.logging:
        os.makedirs('logs', exist_ok=True)
        now = datetime.datetime.now()
        config['logging_file'] = os.path.join('logs', args.dataset + '-' + args.model + '-' + str(
            config['num_labels']) + '-' + now.strftime("%Y%m%d%H%M%S"))
    config['logging'] = args.logging

    # for using k_adpater args in constructing models
    if args.model == 'k_adapter':
        args.device = device
        args.n_gpu = torch.cuda.device_count()
        args.adapter_list = args.adapter_list.split(',')
        args.adapter_list = [int(i) for i in args.adapter_list]
        config['args'] = args
        args.meta_fac_adaptermodel = config['model_archive']['k_adapter_fac']
        args.meta_lin_adaptermodel = config['model_archive']['k_adapter_lin']

    # for ernie to load pre calculated files
    config['tagme_thr'] = args.threshold
    if args.model == 'ernie':
        with open(config['ernie_pre_calculated']['embed'], 'rb') as f:
            embed = pickle.load(f)
            config['ernie_embed'] = embed
        with open(config['ernie_pre_calculated']['ent_map'], 'rb') as f:
            ent_map = pickle.load(f)
            config['ent_map'] = ent_map
        with open(config['ernie_pre_calculated']['entity2id'], 'rb') as f:
            entity2id = pickle.load(f)
            config['entity2id'] = entity2id

    set_seed(config['seed'])

    tokenizer, model = get_tokenizer_and_model(config)
    model.to(device)

    if args.mode == 'train':

        if args.model in ['knowbert-w-w', 'knowbert-wiki', 'knowbert-wordnet']:
            config['batcher'] = CustomKnowBertBatchifier(model_archive=config['model_archive'][config['model_name']],
                                                         batch_size=config['batch_size'])
            train_labels, train_statements = load_dataset(config, 'train')
            val_labels, val_statements = load_dataset(config, 'val')
            trainer = KnowbertTrainer(config, model, train_statements, train_labels, val_statements, val_labels)
        else:
            train_data = NewsDataset(config, tokenizer, 'train')
            val_data = NewsDataset(config, tokenizer, 'val')
            if args.model == 'ernie':
                with open(config['logging_file'] + '_predict.txt', 'a') as f:
                    f.write(
                        f'Total entities in {len(train_data)} training statement in {args.dataset} for threshold at {args.threshold}: {train_data.entity_total}' + "\n")
                    f.write(
                        f'Total entities in {len(val_data)} val statement in {args.dataset} for threshold at {args.threshold}: {val_data.entity_total}' + "\n")

                print(f'Total entities in {len(train_data)} training statement in {args.dataset} for threshold at {args.threshold}: {train_data.entity_total} ')
                print(
                    f'Total entities in {len(val_data)} val statement in {args.dataset} for threshold at {args.threshold}: {val_data.entity_total} ')
            train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data),
                                          batch_size=config['batch_size'])
            val_dataloader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=config['batch_size'])
            trainer = Trainer(config, model, train_dataloader, val_dataloader)

        trainer.train(verbose=config['verbose'])

    else:
        if args.model in ['knowbert-w-w', 'knowbert-wiki', 'knowbert-wordnet']:
            config['batcher'] = CustomKnowBertBatchifier(model_archive=config['model_archive'][config['model_name']],
                                                         batch_size=config['batch_size'])
            test_labels, test_statements = load_dataset(config, 'test')
            predictor = KnowbertPredictor(config, model, test_statements, test_labels)
        else:
            test_data = NewsDataset(config, tokenizer, 'test')
            test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data),
                                         batch_size=config['batch_size'])
            if args.model == 'ernie':
                with open(config['logging_file'] + '_predict.txt', 'a') as f:
                    f.write(
                        f'Total entities in {len(test_data)} test statement in {args.dataset} for threshold at {args.threshold}: {test_data.entity_total}' + "\n")
                print(f'Total entities in {len(test_data)} test statement in {args.dataset} for threshold at {args.threshold}: {test_data.entity_total}' + "\n")

            predictor = Predictor(config, model, test_dataloader)

        predictions, true_labels = predictor.predict()
        if config['num_labels'] == 2:
            label_names = config['covid']['label_names']
        else:
            label_names = config[args.dataset]['label_names']
        report = classification_report(true_labels, predictions, target_names=label_names, digits=4)
        print(report)
        with open(config['logging_file'] + '_predict.txt', 'a') as f:
            if args.model == 'ernie':
                f.write(f"Dataset: {config['dataset_name']}, model: {config['model_name']},"
                        f" batch size: {config['batch_size']}" + f"thr: {config['tagme_thr']}" + "\n")
            else:
                f.write(f"Dataset: {config['dataset_name']}, model: {config['model_name']},"
                        f" batch size: {config['batch_size']}" + "\n")
            f.write(f"{report}" + "\n")
        # report_dict = classification_report(true_labels, predictions, target_names=label_names, digits=4,
        #                                     output_dict=True)
        # save_report(config, report_dict)


if __name__ == "__main__":
    main()

