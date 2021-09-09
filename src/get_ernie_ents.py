"""
Created on 06 Sep 2021
author: Chenxi
"""

import os
import pickle
import torch
import tagme
import pandas as pd
from dataset import load_dataset


def save_sentence_with_ents(statements, labels, thr, ent_map, file_name):
    """
    the most time consuming part of ernie is to annotate and get the entity spans for the same statement for multiple
    runs, so save the list of entities for each statement in a csv file with its label and statement, so can shuffle
    while training. ents = [[ent_map[a.entity_title], a.begin, a.end, a.score]]
    """
    data_list = []
    i = 0
    len_total = len(statements)
    for statement, label in zip(statements, labels):
        ents = get_ents(statement, thr, ent_map)
        data_list.append([label, statement, ents])
        i += 1
        if i % 100 == 0:
            print(f'{i} out of {len_total} done')
    df = pd.DataFrame(data_list, columns=['label', 'statement', 'ents'])
    df.to_csv(file_name)


def get_ents(sentence, thr, ent_map):
    tagme.GCUBE_TOKEN = "c8623405-ea8c-4c06-8394-ef7550483f75-843339462"
    annotation = tagme.annotate(sentence)

    ents = []
    for a in annotation.get_annotations(thr):
        if a.entity_title not in ent_map:
            continue
        ents.append([ent_map[a.entity_title], a.begin, a.end, a.score])
    return ents


def get_embed(path):
    """Altogether will out of memory so we split to two parts. we get the tensor separately and combine them before
     converting to one embedding"""

    vecs = [[0] * 100]
    with open(path + "/entity2vec.vec", 'r') as fin:
        a = 1
        for line in fin:
            if a < 3000000:
                vec = line.strip().split('\t')
                vec = [float(x) for x in vec]
                vecs.append(vec)
            a += 1
    embed = torch.FloatTensor(vecs)
    # embed = torch.nn.Embedding.from_pretrained(embed)
    with open('tensor_embed1.txt', 'wb') as f:
        pickle.dump(embed, f)

    vecs = [[0] * 100]
    with open(path + "/entity2vec.vec", 'r') as fin:
        a = 1
        for line in fin:
            if a >= 3000000:
                vec = line.strip().split('\t')
                vec = [float(x) for x in vec]
                vecs.append(vec)
            a += 1
    embed = torch.FloatTensor(vecs)
    # embed = torch.nn.Embedding.from_pretrained(embed)
    with open('tensor_embed2.txt', 'wb') as f:
        pickle.dump(embed, f)

    with open('tensor_embed1.txt', 'rb') as f:
        tensor_e1 = pickle.load(f)

    with open('tensor_embed2.txt', 'rb') as f:
        tensor_e2 = pickle.load(f)

    '''now we concatenate the two tensors along the 0-dimension'''
    embed = torch.cat((tensor_e1, tensor_e2), 0)
    embed = torch.nn.Embedding.from_pretrained(embed)
    with open('embed.txt', 'wb') as f:
        pickle.dump(embed, f)


def save_ents(config):
    train_labels, train_statements = load_dataset(config, 'train')
    val_labels, val_statements = load_dataset(config, 'val')

    test_labels, test_statements = load_dataset(config, 'test')
    print('==========================================================================')
    print(f"here I am starting to work on {config['dataset_name']} train dataset")
    save_sentence_with_ents(train_statements, train_labels, 0, config['ent_map'],
                            os.path.join('../data/ents_csv', config['dataset_name'] + '_' + 'train' + '.csv'))
    print('')
    print('==========================================================================')
    print(f"here I am starting to work on {config['dataset_name']} val dataset")
    save_sentence_with_ents(val_statements, val_labels, 0, config['ent_map'],
                            os.path.join('../data/ents_csv', config['dataset_name'] + '_' + 'val' + '.csv'))
    print('')
    print('==========================================================================')
    print(f"here I am starting to work on {config['dataset_name']} test dataset")
    save_sentence_with_ents(test_statements, test_labels, 0, config['ent_map'],
                            os.path.join('../data/ents_csv', config['dataset_name'] + '_' + 'test' + '.csv'))


def marge_ents_to_dataset(original_data_df, df_with_ents, file_name):
    original_data_df['ents'] = df_with_ents['ents']
    original_data_df.to_csv(file_name)

