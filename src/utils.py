"""
Created on 02 Sep 2021
author: Chenxi
"""

from datetime import datetime
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
    report_path = os.path.join(config['report_path'], config['dataset_name'], config['model_name'], str(config['num_labels']))
    os.makedirs(report_path, exist_ok=True)
    report_df = pd.DataFrame(report_dict).transpose()
    now = datetime.datetime.now()
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


def to_device(obj, device):
    """
    Given a structure (possibly) containing Tensors on the CPU,
    move all the Tensors to the specified GPU (or do nothing, if they should be on the CPU).
    """

    if device == torch.device("cpu"):
        return obj
    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {key: to_device(value, device) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [to_device(item, device) for item in obj]
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # This is the best way to detect a NamedTuple, it turns out.
        return obj.__class__(*(to_device(item, device) for item in obj))
    elif isinstance(obj, tuple):
        return tuple(to_device(item, device) for item in obj)
    else:
        return obj
