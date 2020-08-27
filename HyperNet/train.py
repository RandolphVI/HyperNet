# -*- coding:utf-8 -*-
__author__ = 'randolph'

import os
import sys
import time
import torch
import torchsnooper
import torch.nn as nn

sys.path.append('../')

from layers import MOOCNet, Loss
from utils import checkmate as cm
from utils import data_helpers as dh
from utils import param_parser as parser
from tqdm import tqdm, trange
import torch.nn.utils.rnn as rnn_utils
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score


args = parser.parameter_parser()
OPTION = dh.option()
logger = dh.logger_fn("ptlog", "logs/{0}-{1}.log".format('Train' if OPTION == 'T' else 'Restore', time.asctime()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_input_data(record):
    """
    Creating features and targets with Torch tensors.
    """
    x_activity, x_lens, x_tsp, y = record
    batch_x_pack = rnn_utils.pack_padded_sequence(x_activity, x_lens, batch_first=True)
    batch_x_pack = batch_x_pack.to(device)
    y = y.type(torch.FloatTensor).to(device)
    return batch_x_pack, x_tsp, y


def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.1)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')


def print_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            print("weight", m.weight.data)
            print("bias:", m.bias.data)
            print("next...")


def train():
    """Training QuesNet model."""
    dh.tab_printer(args, logger)

    # Load sentences, labels, and training parameters
    logger.info("Loading data...")
    logger.info("Data processing...")
    train_data = dh.load_data_and_labels(args.train_file)
    val_data = dh.load_data_and_labels(args.validation_file)

    logger.info("Data padding...")
    train_dataset = dh.MyData(train_data.activity, train_data.timestep, train_data.labels)
    val_dataset = dh.MyData(val_data.activity, val_data.timestep, val_data.labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dh.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dh.collate_fn)

    # Load word2vec model
    COURSE_SIZE = dh.course2vec(args.course2vec_file)

    # Init network
    logger.info("Init nn...")
    net = MOOCNet(args, COURSE_SIZE).to(device)

    # weights_init(model=net)
    # print_weight(model=net)

    print("Model's state_dict:")
    for param_tensor in net.state_dict():
        print(param_tensor, "\t", net.state_dict()[param_tensor].size())

    criterion = Loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.l2_lambda)

    if OPTION == 'T':
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        saver = cm.BestCheckpointSaver(save_dir=out_dir, num_to_keep=args.num_checkpoints, maximize=False)
        logger.info("Writing to {0}\n".format(out_dir))
    elif OPTION == 'R':
        timestamp = input("[Input] Please input the checkpoints model you want to restore: ")
        while not (timestamp.isdigit() and len(timestamp) == 10):
            timestamp = input("[Warning] The format of your input is illegal, please re-input: ")
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        saver = cm.BestCheckpointSaver(save_dir=out_dir, num_to_keep=args.num_checkpoints, maximize=False)
        logger.info("Writing to {0}\n".format(out_dir))
        checkpoint = torch.load(out_dir)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    logger.info("Training...")
    writer = SummaryWriter('summary')

    def eval_model(val_loader, epoch):
        """
        Evaluate on the validation set.
        """
        net.eval()
        eval_loss = 0.0
        true_labels, predicted_scores, predicted_labels = [], [], []
        for batch in val_loader:
            x_val, tsp_val, y_val = create_input_data(batch)
            logits, scores = net(x_val, tsp_val)
            avg_batch_loss = criterion(scores, y_val)
            eval_loss = eval_loss + avg_batch_loss.item()
            for i in y_val.tolist():
                true_labels.append(i)
            for j in scores.tolist():
                predicted_scores.append(j)
                if j >= args.threshold:
                    predicted_labels.append(1)
                else:
                    predicted_labels.append(0)

        # Calculate the Metrics
        eval_acc = accuracy_score(true_labels, predicted_labels)
        eval_pre = precision_score(true_labels, predicted_labels)
        eval_rec = recall_score(true_labels, predicted_labels)
        eval_F1 = f1_score(true_labels, predicted_labels)
        eval_auc = roc_auc_score(true_labels, predicted_scores)
        eval_prc = average_precision_score(true_labels, predicted_scores)
        eval_loss = eval_loss / len(val_loader)
        cur_value = eval_F1
        logger.info("All Validation set: Loss {0:g} | ACC {1:.4f} | PRE {2:.4f} | REC {3:.4f} | F1 {4:.4f} | AUC {5:.4f} | PRC {6:.4f}"
                    .format(eval_loss, eval_acc, eval_pre, eval_rec, eval_F1, eval_auc, eval_prc))
        writer.add_scalar('validation loss', eval_loss, epoch)
        writer.add_scalar('validation ACC', eval_acc, epoch)
        writer.add_scalar('validation PRECISION', eval_pre, epoch)
        writer.add_scalar('validation RECALL', eval_rec, epoch)
        writer.add_scalar('validation F1', eval_F1, epoch)
        writer.add_scalar('validation AUC', eval_auc, epoch)
        writer.add_scalar('validation PRC', eval_prc, epoch)
        return cur_value

    for epoch in tqdm(range(args.epochs), desc="Epochs:", leave=True):
        # Training step
        batches = trange(len(train_loader), desc="Batches", leave=True)
        for batch_cnt, batch in zip(batches, train_loader):
            net.train()
            x_train, tsp_train, y_train = create_input_data(batch)
            optimizer.zero_grad()   # 如果不置零，Variable 的梯度在每次 backward 的时候都会累加
            logits, scores = net(x_train, tsp_train)
            # TODO
            avg_batch_loss = criterion(scores, y_train)
            avg_batch_loss.backward()
            optimizer.step()    # Parameter updating
            batches.set_description("Batches (Loss={:.4f})".format(avg_batch_loss.item()))
            logger.info('[epoch {0}, batch {1}] loss: {2:.4f}'.format(epoch + 1, batch_cnt, avg_batch_loss.item()))
            writer.add_scalar('training loss', avg_batch_loss, batch_cnt)
        # Evaluation step
        cur_value = eval_model(val_loader, epoch)
        saver.handle(cur_value, net, optimizer, epoch)
    writer.close()

    logger.info('Training Finished.')


if __name__ == "__main__":
    train()
