# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import os
import math
import gensim
import logging
import json
import torch
import numpy as np
import pandas as pd

from scipy import stats
from texttable import Texttable
from gensim.models import KeyedVectors
import torch.nn.utils.rnn as rnn_utils


def option():
    """
    Choose training or restore pattern.

    Returns:
        The OPTION
    """
    OPTION = input("[Input] Train or Restore? (T/R): ")
    while not (OPTION.upper() in ['T', 'R']):
        OPTION = input("[Warning] The format of your input is illegal, please re-input: ")
    return OPTION.upper()


def logger_fn(name, input_file, level=logging.INFO):
    """
    The Logger.

    Args:
        name: The name of the logger
        input_file: The logger file path
        level: The logger level
    Returns:
        The logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    log_dir = os.path.dirname(input_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File Handler
    fh = logging.FileHandler(input_file, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # stream Handler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(logging.WARNING)
    logger.addHandler(sh)
    return logger


def tab_printer(args, logger):
    """
    Function to print the logs in a nice tabular format.

    Args:
        args: Parameters used for the model.
        logger: The logger
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    t.add_rows([["Parameter", "Value"]])
    logger.info('\n' + t.draw())


def get_model_name():
    """
    Get the model name used for test.

    Returns:
        The model name
    """
    MODEL = input("[Input] Please input the model file you want to test, it should be like (1490175368): ")

    while not (MODEL.isdigit() and len(MODEL) == 10):
        MODEL = input("[Warning] The format of your input is illegal, "
                      "it should be like (1490175368), please re-input: ")
    return MODEL


def create_prediction_file(save_dir, identifiers, predictions):
    """
    Create the prediction file.

    Args:
        save_dir: The all classes predicted results provided by network
        identifiers: The data record id
        predictions: The predict scores
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    preds_file = os.path.abspath(os.path.join(save_dir, 'submission.json'))
    with open(preds_file, 'w') as fout:
        tmp_dict = {}
        for index, predicted_label in enumerate(predictions):
            if identifiers[index] not in tmp_dict:
                tmp_dict[identifiers[index]] = [predicted_label]
            else:
                tmp_dict[identifiers[index]].append(predicted_label)

        for key in tmp_dict.keys():
            data_record = {
                'item_id': key,
                'label_list': tmp_dict[key],
            }
            fout.write(json.dumps(data_record, ensure_ascii=False) + '\n')


def evaluation(true_label, pred_label):
    """
    Calculate the PCC & DOA.

    Args:
        true_label: The true labels
        pred_label: The predicted labels
    Returns:
        The value of PCC & DOA
    """
    # compute pcc
    pcc, _ = stats.pearsonr(pred_label, true_label)
    if math.isnan(pcc):
        print('[Error]: PCC=nan', true_label, pred_label)
    # compute doa
    n = 0
    correct_num = 0
    for i in range(len(true_label) - 1):
        for j in range(i + 1, len(true_label)):
            if (true_label[i] > true_label[j]) and (pred_label[i] > pred_label[j]):
                correct_num += 1
            elif (true_label[i] == true_label[j]) and (pred_label[i] == pred_label[j]):
                continue
            elif (true_label[i] < true_label[j]) and (pred_label[i] < pred_label[j]):
                correct_num += 1
            n += 1
    if n == 0:
        print(true_label)
        return -1, -1
    doa = correct_num / n
    return pcc, doa


def course2vec(course2idx_file):
    """
    Return the word2vec model matrix.

    Args:
        course2idx_file: The course2idx file
    Returns:
        The word2vec model matrix
    Raises:
        IOError: If word2vec model file doesn't exist
    """
    if not os.path.isfile(course2idx_file):
        raise IOError("[Error] The word2vec file doesn't exist. ")

    with open(course2idx_file, 'r') as handle:
        course2idx = json.load(handle)

    course_cnt = len(course2idx)
    return course_cnt


def load_data_and_labels(input_file):
    if not input_file.endswith('.json'):
        raise IOError("[Error] The research data is not a json file. "
                      "Please preprocess the research data into the json file.")
    with open(input_file) as fin:
        id_list = []
        activity_list = []
        timestep_list = []
        labels_list = []

        for index, eachline in enumerate(fin):
            data = json.loads(eachline)
            id = data['item_id']
            activity = data['activity']
            timestep = data['timestep']
            labels = data['labels']

            id_list.append(id)
            activity_list.append(activity)
            timestep_list.append(timestep)
            labels_list.append(labels)

    class _Data:
        def __init__(self):
            pass

        @property
        def id(self):
            return id_list

        @property
        def activity(self):
            return activity_list

        @property
        def timestep(self):
            return timestep_list

        @property
        def labels(self):
            return labels_list

    return _Data()


class MyData(torch.utils.data.Dataset):
    """
    定义数据读取迭代器结构
    """
    def __init__(self, data_seq, data_tsp, data_label):
        self.seqs = data_seq
        self.tsp = data_tsp
        self.labels = data_label

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.tsp[idx], self.labels[idx]


def collate_fn(data):
    """
    Version for PyTorch

    Args:
        data: The research data. 0-dim: word token index / 1-dim: data label
    Returns:
        pad_content: The padded data
        lens: The ground truth lengths
        labels: The data labels
    """
    data.sort(key=lambda x: len(x[0]), reverse=True)
    data_lens = [len(i[0]) for i in data]
    data_x = [torch.tensor(i[0]) for i in data]
    data_tsp = [i[1] for i in data]
    data_y = torch.tensor([i[2] for i in data])
    pad_content = rnn_utils.pad_sequence(data_x, batch_first=True, padding_value=0.)
    return pad_content.unsqueeze(-1), data_lens, data_tsp, data_y