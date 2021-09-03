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
# import torch
# import random
# import numpy as np
# import pandas as pd
import yaml
import yamlordereddictloader
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import classification_report
from kb.knowbert_utils import KnowBertBatchifier
# from allennlp.models.archival import load_archive
# from allennlp.data import Vocabulary
from dataset import NewsDataset
from trainer import Trainer
from knowbert_trainer import KnowbertTrainer
from predictor import Predictor, KnowbertPredictor
from model_loader import get_tokenizer_and_model
from utils import *


#
#
# def save_report(config, report_dict):
#     report_path = os.path.join(config['report_path'], config['dataset_name'], config['model_name'])
#     os.makedirs(report_path, exist_ok=True)
#     report_df = pd.DataFrame(report_dict).transpose()
#     now = datetime.now()
#     date_time = now.strftime("%Y%m%d%H%M%S")
#     report_file_name = os.path.join(report_path, date_time + '.txt')
#     report_df.to_csv(report_file_name)


# def set_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
#     np.random.seed(seed)              # Numpy module
#     random.seed(seed)                 # Python random module
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='covid', choices=['liar', 'covid'])
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--model', type=str, default='roberta-base')
    parser.add_argument('--verbose', action='store_true', default=False, help='print detailed training process')

    # k_adapters
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
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='covid', choices=['liar', 'covid'])
    # parser.add_argument('--config', type=str, default='config.yaml')
    # parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    # parser.add_argument('--model', type=str, default='roberta-base')
    # parser.add_argument('--verbose', action='store_true', default=False, help='print detailed training process')
    #
    # # k_adapters
    # parser.add_argument("--output_dir", default='./proc/roberta_adapter', type=str,
    #                     help="The output directory where the model predictions and checkpoints will be written.")
    # parser.add_argument("--freeze_bert", default=False, type=bool,
    #                     help="freeze the parameters of original model.")
    # parser.add_argument("--freeze_adapter", default=True, type=bool,
    #                     help="freeze the parameters of adapter.")
    # parser.add_argument('--fusion_mode', type=str, default='concat',
    #                     help='the fusion mode for bert feature and adapter feature |add|concat')
    #
    # parser.add_argument("--adapter_transformer_layers", default=2, type=int,
    #                     help="The transformer layers of adapter.")
    # parser.add_argument("--adapter_size", default=768, type=int,
    #                     help="The hidden size of adapter.")
    # parser.add_argument("--adapter_list", default="0,11,23", type=str,
    #                     help="The layer where add an adapter")
    # parser.add_argument("--adapter_skip_layers", default=0, type=int,
    #                     help="The skip_layers of adapter according to bert layers")
    # parser.add_argument("--no_cuda", default=False, action='store_true',
    #                     help="Avoid using CUDA when available")
    # parser.add_argument('--meta_fac_adaptermodel', default="../pretrained_models/fac-adapter/pytorch_model.bin",
    #                     type=str, help='the pretrained factual adapter model')
    # parser.add_argument('--meta_lin_adaptermodel', default="../pretrained_models/lin-adapter/pytorch_model.bin",
    #                     type=str, help='the pretrained linguistic adapter model')

    args = parse_args()
    args.model = 'k_adapters'


    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yamlordereddictloader.Loader)

    config = dict(config)
    device = torch.device("cuda:{}".format(config['gpu']) if config['gpu'] != -1 else "cpu")
    config['device'] = device
    config['model_name'] = args.model
    config['dataset_name'] = args.dataset
    config['verbose'] = args.verbose

    # for using k_adpaters args in constructing models
    args.device = device
    args.n_gpu = torch.cuda.device_count()
    args.adapter_list = args.adapter_list.split(',')
    args.adapter_list = [int(i) for i in args.adapter_list]
    config['args'] = args

    args.meta_fac_adaptermodel = config['model_archive']['k_adapters_fac']
    args.meta_lin_adaptermodel = config['model_archive']['k_adapters_lin']

    set_seed(config['seed'])



    tokenizer, model = get_tokenizer_and_model(config)
    model.to(device)

    batcher = None
    if args.model in ['knowbert-w-w', 'knowbert-wiki', 'knowbert-wordnet']:
        model_archive = config['model_archive']['model_name']
        batcher = KnowBertBatchifier(model_archive)

    if args.mode == 'train':
        train_data = NewsDataset(config, args.dataset, tokenizer, 'train', batcher)
        val_data = NewsDataset(config, args.dataset, tokenizer, 'val', batcher)
        train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=config['batch_size'])
        val_dataloader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=config['batch_size'])

        if args.model in ['knowbert-w-w', 'knowbert-wiki', 'knowbert-wordnet']:
            trainer = KnowbertTrainer(config, model, train_dataloader, val_dataloader)
        else:  # up to now, all the bert, roberta and k-adapters use the standard Trainer
            trainer = Trainer(config, model, train_dataloader, val_dataloader)
        trainer.train(verbose=config['verbose'])
    else:
        test_data = NewsDataset(config, args.dataset, tokenizer, 'test', batcher)
        test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=config['batch_size'])

        if args.model in ['knowbert-w-w', 'knowbert-wiki', 'knowbert-wordnet']:
            predictor = KnowbertPredictor(config, model, test_dataloader)  # implement!!
        else:  # up to now, all the bert, roberta and k-adapters use the standard Trainer
            predictor = Predictor(config, model, test_dataloader)
        predictions, true_labels = predictor.predict()
        label_names = config[args.dataset]['class_names']
        report = classification_report(true_labels, predictions, target_names=label_names, digits=4)
        print(report)
        report_dict = classification_report(true_labels, predictions, target_names=label_names, digits=4,
                                            output_dict=True)
        save_report(config, report_dict)


if __name__ == "__main__":
    main()
    #
    # from kb.include_all import ModelArchiveFromParams
    # from kb.knowbert_utils import KnowBertBatchifier
    # from allennlp.common import Params
    #
    # import torch
    #
    # # a pretrained model, e.g. for Wordnet+Wikipedia
    # archive_file = 'https://allennlp.s3-us-west-2.amazonaws.com/knowbert/models/knowbert_wiki_wordnet_model.tar.gz'
    #
    # # load model and batcher
    # params = Params({"archive_file": archive_file})
    # model = ModelArchiveFromParams.from_params(params=params)
    # batcher = KnowBertBatchifier(archive_file)
    #
    # sentences = ["Paris is located in France.", "KnowBert is a knowledge enhanced BERT"]
    #
    # # batcher takes raw untokenized sentences
    # # and yields batches of tensors needed to run KnowBert
    # for batch in batcher.iter_batches(sentences, verbose=True):
    #     # model_output['contextual_embeddings'] is (batch_size, seq_len, embed_dim) tensor of top layer activations
    #     model_output = model(**batch)
    #     for key, value in batch.items():
    #         print('hahahahahahhahahahahahah')
    #         print("%s == %s" % (key, value))
    #
    # a = 1

    # from kb.include_all import ModelArchiveFromParams
    # from allennlp.common import Params
    #
    # archive_file = 'https://allennlp.s3-us-west-2.amazonaws.com/knowbert/models/knowbert_wiki_wordnet_model.tar.gz'
    # archive_file = '../knowbert_models/knowbert_wiki_wordnet_model.tar.gz'
    #
    # # load model and batcher
    # params = Params({"archive_file": archive_file})
    # model1 = ModelArchiveFromParams.from_params(params=params)
    #
    #
    #
    # archive_model = load_archive(archive_file)
    #
    # # load knowbert model and vocabulary
    # knowbert_model = archive_model.model
    # vocab = Vocabulary.from_params(archive_model.config['vocabulary'])
    # a = 1

    # from allennlp.models.archival import load_archive
    # from allennlp.data import DatasetReader, Vocabulary, DataIterator
    # from allennlp.nn.util import move_to_device
    # from allennlp.common import Params
    #
    # import numpy as np
    #
    # from kb.include_all import *
    #
    #
    # def write_for_official_eval():
    #     model_archive_file = '../knowbert_models/knowbert_wiki_wordnet_model.tar.gz'
    #     archive = load_archive(model_archive_file)
    #     model = archive.model
    #
    #     reader = DatasetReader.from_params(archive.config['dataset_reader'])
    #
    #     iterator = DataIterator.from_params(Params({"type": "basic", "batch_size": 32}))
    #     vocab = Vocabulary.from_params(archive.config['vocabulary'])
    #     iterator.index_with(vocab)
    #
    #     model.cuda()
    #     model.eval()
    #
    #     label_ids_to_label = {0: 'F', 1: 'T'}
    #
    #     instances = reader.read('tests/fixtures/evaluation/wic' + '/train')
    #     predictions = []
    #     for batch in iterator(instances, num_epochs=1, shuffle=False):
    #         batch = move_to_device(batch, cuda_device=0)
    #         output = model(**batch)
    #         for key, value in batch.items():
    #             print('hahahahahahhahahahahahah')
    #             print("%s == %s" % (key, value))
    #
    #         batch_labels = [
    #             label_ids_to_label[i]
    #             for i in output['predictions'].cpu().numpy().tolist()
    #         ]
    #
    #         predictions.extend(batch_labels)
    #
    #
    # write_for_official_eval()
    # a = 1
