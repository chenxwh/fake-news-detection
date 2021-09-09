"""
Created on 02 Sep 2021
author: Chenxi
"""

import os
import torch
import numpy as np
from utils import *


class Predictor:
    def __init__(self, config, model, test_dataloader):
        if config['model_name'] == 'ernie':
            model_path = os.path.join(config['save_path'], config['dataset_name'] + '-' + config['model_name']
                                      + '-' + str(config['num_labels']) + '-thr-' + str(config['tagme_thr']) + '.pth')
        else:
            model_path = os.path.join(config['save_path'], config['dataset_name'] + '-' + config['model_name'] +
                                      '-' + str(config['num_labels']) + '.pth')
        assert os.path.exists(model_path)
        self.model = model
        self.model.load_state_dict(torch.load(model_path))
        self.config = config
        self.test_dataloader = test_dataloader

    def predict(self):
        predictions, true_labels = [], []
        self.model.eval()
        for batch in self.test_dataloader:
            with torch.no_grad():

                if self.config['model_name'] == 'ernie':

                    embed = self.config['ernie_embed']
                    b_input_ids = batch['input_ids'].to(self.config['device'])
                    b_input_ent = embed(batch['input_ent'] + 1).to(self.config['device'])
                    b_ent_mask = batch['ent_mask'].to(self.config['device'])

                    result = self.model(b_input_ids,
                                        input_ent=b_input_ent,
                                        ent_mask=b_ent_mask)

                    logits = result.logits

                else:
                    b_input_ids = batch['input_ids'].to(self.config['device'])
                    b_input_mask = batch['attention_mask'].to(self.config['device'])

                    if self.config['model_name'] in ['k_adapter']:
                        logits = self.model(b_input_ids,
                                            attention_mask=b_input_mask)
                    else:
                        result = self.model(b_input_ids,
                                            attention_mask=b_input_mask,
                                            return_dict=True)

                        logits = result.logits

            b_labels = batch['labels'].to(self.config['device'])

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            pred_flat = np.argmax(logits, axis=1).flatten()
            labels_flat = label_ids.flatten()

            # Store predictions and true labels
            for pred in pred_flat:
                predictions.append(pred)
            for label in labels_flat:
                true_labels.append(label)

        return predictions, true_labels


class KnowbertPredictor:
    def __init__(self, config, model, test_statements, test_labels):
        model_path = os.path.join(config['save_path'], config['dataset_name'] + '-' + config['model_name'] +
                                  '-' + str(config['num_labels']) + '.pth')
        assert os.path.exists(model_path)
        self.model = model
        self.model.load_state_dict(torch.load(model_path))
        self.config = config
        self.test_statements = test_statements
        self.test_labels = test_labels

    def predict(self):
        predictions, true_labels = [], []
        self.model.eval()
        for batch in self.config['batcher'].iter_batches(self.test_statements, self.test_labels):
            tokens = to_device(batch['tokens'], self.config['device'])
            segment_ids = to_device(batch['segment_ids'], self.config['device'])
            candidates = to_device(batch['candidates'], self.config['device'])
            label_ids = to_device(batch['label_ids'], self.config['device'])

            with torch.no_grad():
                result = self.model(tokens=tokens,
                                    segment_ids=segment_ids,
                                    candidates=candidates,
                                    label_ids=label_ids)

            logits = result.logits

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()

            pred_flat = np.argmax(logits, axis=1).flatten()
            labels_flat = label_ids.flatten()

            # Store predictions and true labels
            for pred in pred_flat:
                predictions.append(pred)
            for label in labels_flat:
                true_labels.append(label)

        return predictions, true_labels
