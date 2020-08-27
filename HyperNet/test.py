# -*- coding:utf-8 -*-
__author__ = 'randolph'

import os
import sys
import time
import torch

sys.path.append('../')

from layers import MOOCNet, Loss
from utils import checkmate as cm
from utils import data_helpers as dh
from utils import param_parser as parser
from tqdm import tqdm, trange
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score


args = parser.parameter_parser()
MODEL = dh.get_model_name()
logger = dh.logger_fn("ptlog", "logs/Test-{0}.log".format(time.asctime()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CPT_DIR = os.path.abspath(os.path.join(os.path.curdir, "runs", MODEL))
SAVE_DIR = os.path.abspath(os.path.join(os.path.curdir, "outputs", MODEL))


def create_input_data(record):
    """
    Creating features and targets with Torch tensors.
    """
    x_activity, x_lens, x_tsp, y = record
    batch_x_pack = rnn_utils.pack_padded_sequence(x_activity, x_lens, batch_first=True)
    batch_x_pack = batch_x_pack.to(device)
    return batch_x_pack, x_tsp, y


def test():
    logger.info("Loading Data...")
    logger.info("Data processing...")

    test_data = dh.load_data_and_labels(args.test_file)

    test_dataset = dh.MyData(test_data.activity, test_data.timestep, test_data.labels)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dh.collate_fn)

    # Load word2vec model
    COURSE_SIZE = dh.course2vec(args.course2vec_file)

    criterion = Loss()
    net = MOOCNet(args, COURSE_SIZE).to(device)
    checkpoint_file = cm.get_best_checkpoint(CPT_DIR, select_maximum_value=False)
    checkpoint = torch.load(checkpoint_file)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    logger.info("Scoring...")
    true_labels, predicted_scores, predicted_labels = [], [], []
    batches = trange(len(test_loader), desc="Batches", leave=True)
    for batch_cnt, batch in zip(batches, test_loader):
        x_test, tsp_test, y_test = create_input_data(batch)
        logits, scores = net(x_test, tsp_test)
        for i in y_test.tolist():
            true_labels.append(i)
        for j in scores.tolist():
            predicted_scores.append(j)
            if j >= 0.5:
                predicted_labels.append(1)
            else:
                predicted_labels.append(0)

    # Calculate the Metrics
    logger.info('Test Finished.')

    logger.info('Creating the prediction file...')
    dh.create_prediction_file(save_dir=SAVE_DIR, identifiers=test_data.id, predictions=predicted_labels)

    logger.info('All Finished.')


if __name__ == "__main__":
    test()

