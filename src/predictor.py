"""
Created on 02 Sep 2021
author: Chenxi
"""


import os
import torch
import numpy as np


class Predictor:
    def __init__(self, config, model, test_dataloader):
        model_path = os.path.join(config['save_path'], config['dataset_name'] + '-' + config['model_name'] + '.pth')
        assert os.path.exists(model_path)
        self.model = model
        self.model.load_state_dict(torch.load(model_path))
        self.config = config
        self.test_dataloader = test_dataloader

    def predict(self):
        predictions, true_labels = [], []
        self.model.eval()
        for batch in self.test_dataloader:
            b_input_ids = batch['input_ids'].to(self.config['device'])
            b_input_mask = batch['attention_mask'].to(self.config['device'])
            b_labels = batch['labels'].to(self.config['device'])

            with torch.no_grad():
                # Forward pass, calculate logit predictions, no labels!
                result = self.model(b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask,
                                    return_dict=True)

            logits = result.logits

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
