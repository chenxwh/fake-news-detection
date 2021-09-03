"""
Created on 02 Sep 2021
author: Chenxi
"""

import datetime
import numpy as np
import pandas as pd
import os
import torch
import random


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round(elapsed))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def flat_accuracy(preds, labels):
    """
    Function to calculate the accuracy of our predictions vs labels
    """
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


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