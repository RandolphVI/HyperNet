# -*- coding:utf-8 -*-
__author__ = 'randolph'

"""MOOCNet layers."""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import weight_norm
import torch.nn.utils.rnn as rnn_utils


class ConvLayer(nn.Module):
    def __init__(self, input_units, output_units, filter_size, padding_sizes, dropout=0.2):
        super(ConvLayer, self).__init__()
        self.conv = weight_norm(nn.Conv1d(in_channels=input_units, out_channels=output_units,
                                          kernel_size=filter_size[0], stride=1, padding=padding_sizes[0]))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv, self.relu, self.dropout)
        self.init_weights()

    def init_weights(self):
        self.conv.weight.data.normal_(0, 0.01)

    def forward(self, input_x):
        """
        Convolution Layer.
        Args:
            input_x: [batch_size, embedding_size, sequence_length]
        Returns:
            conv_out: [batch_size, sequence_length, num_filters]
            conv_avg: [batch_size, num_filters]
        """
        conv_out = self.net(input_x)

        conv_out = conv_out.permute(0, 2, 1)
        conv_avg = torch.mean(conv_out, dim=1)
        return conv_out, conv_avg


class BiRNNLayer(nn.Module):
    def __init__(self, input_units, rnn_type, rnn_layers, rnn_hidden_size, dropout_keep_prob):
        super(BiRNNLayer, self).__init__()
        if rnn_type == 'LSTM':
            self.bi_rnn = nn.LSTM(input_size=input_units, hidden_size=rnn_hidden_size, num_layers=rnn_layers,
                                  batch_first=True, bidirectional=False, dropout=dropout_keep_prob)
        if rnn_type == 'GRU':
            self.bi_rnn = nn.GRU(input_size=input_units, hidden_size=rnn_hidden_size, num_layers=rnn_layers,
                                 batch_first=True, bidirectional=True, dropout=dropout_keep_prob)

    def forward(self, input_x, input_lens):
        """
        RNN Layer.
        Args:
            input_x: [batch_size, batch_max_seq_len, num_filters]
            input_lens: The ground truth lengths of each sequence
        Returns:
            rnn_out: [batch_size, batch_max_seq_len, rnn_hidden_size]
            rnn_avg: [batch_size, num_filters]
        """
        rnn_out, _ = self.bi_rnn(input_x)
        temp = []
        batch_size = input_x.size()[0]
        for i in range(batch_size):
            last_tsp_state = rnn_out[i, input_lens[i] - 1, :]
            temp.append(last_tsp_state)
        temp = torch.stack(temp, 0)
        return temp


class SkipRNNLayer(nn.Module):
    def __init__(self, input_units, rnn_type, rnn_layers, skip_hidden_size, num_filters, skip_size, dropout_keep_prob):
        super(SkipRNNLayer, self).__init__()
        self.num_filters = num_filters
        self.skip_size = skip_size
        self.skip_hidden_size = skip_hidden_size
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=input_units, hidden_size=skip_hidden_size, num_layers=rnn_layers,
                               batch_first=False, bidirectional=False, dropout=dropout_keep_prob)
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=input_units, hidden_size=skip_hidden_size, num_layers=rnn_layers,
                              batch_first=False, bidirectional=False, dropout=dropout_keep_prob)
        self.dropout = nn.Dropout(dropout_keep_prob)

    def forward(self, input_x):
        """
        Skip-RNN Layer.
        Args:
            input_x: [batch_size, sequence_length, num_filters]
        Returns:
            skip_rnn_out: [batch_size, skip_size * skip_hidden_size]
        """
        seq_len = list(input_x.size())[-2]
        pt = seq_len // self.skip_size

        # input_x_trans: [?, num_filters, seq_len]
        input_x_trans = input_x.permute(0, 2, 1)
        s = input_x_trans[:, :, int(-pt * self.skip_size):].contiguous()

        # [?, num_filters, pt, skip_size]
        s = s.view(-1, self.num_filters, pt, self.skip_size)
        # [pt, ?, skip_size, num_filters]
        s = s.permute(2, 0, 3, 1).contiguous()
        # [pt, ? * skip_size, num_filters]
        s = s.view(pt, -1, self.num_filters)

        # skip_rnn_out: [1, ? * skip_size, skip_hidden_size]
        _, skip_rnn_out = self.rnn(s)
        # [?, skip_size * skip_hidden_size]
        skip_rnn_out = skip_rnn_out.view(-1, self.skip_size * self.skip_hidden_size)
        skip_rnn_out = self.dropout(skip_rnn_out)
        return skip_rnn_out


class HighwayLayer(nn.Module):
    def __init__(self, in_units, out_units):
        super(HighwayLayer, self).__init__()
        self.highway_linear = nn.Linear(in_features=in_units, out_features=out_units, bias=True)
        self.highway_gate = nn.Linear(in_features=in_units, out_features=out_units, bias=True)

    def forward(self, input_x):
        highway_g = torch.relu(self.highway_linear(input_x))
        highway_t = torch.sigmoid(self.highway_gate(input_x))
        highway_out = torch.mul(highway_g, highway_t) + torch.mul((1 - highway_t), input_x)
        return highway_out


class MOOCNet(nn.Module):
    """An implementation of QuesNet"""
    def __init__(self, args, course_size, pretrained_embedding=None):
        super(MOOCNet, self).__init__()
        """
        :param args: Arguments object.
        """
        self.args = args
        self.course_size = course_size
        self.pretrained_embedding = pretrained_embedding
        self._setup_layers()

    def _setup_embedding_layer(self):
        """
        Creating Embedding layers.
        """
        if self.pretrained_embedding is None:
            embedding_weight = torch.FloatTensor(np.random.uniform(-1, 1, size=(self.course_size, self.args.embedding_dim)))
            embedding_weight = Variable(embedding_weight, requires_grad=True)
        else:
            if self.args.embedding_type == 0:
                embedding_weight = torch.from_numpy(self.pretrained_embedding).float()
            if self.args.embedding_type == 1:
                embedding_weight = Variable(torch.from_numpy(self.pretrained_embedding).float(), requires_grad=True)
        self.embedding = nn.Embedding(self.course_size, self.args.embedding_dim, _weight=embedding_weight)

    def _setup_conv_layer(self):
        """
        Creating Convolution Layer.
        """
        # TODO
        self.conv = ConvLayer(input_units=self.embedding_dim, output_units=self.args.num_filters[0],
                              filter_size=self.args.filter_sizes, padding_sizes=self.args.conv_padding_sizes)

    def _setup_bi_rnn_layer(self):
        """
        Creating Bi-RNN Layer.
        """
        self.bi_rnn = BiRNNLayer(input_units=self.args.embedding_dim, rnn_type=self.args.rnn_type,
                                 rnn_layers=self.args.rnn_layers, rnn_hidden_size=self.args.rnn_dim,
                                 dropout_keep_prob=self.args.dropout_rate)

    def _setup_skip_rnn_layer(self):
        """
        Creating Skip-RNN Layer.
        """
        self.skip_rnn = SkipRNNLayer(input_units=self.args.num_filters[0], rnn_type=self.args.rnn_type,
                                     rnn_layers=self.args.rnn_layers, skip_hidden_size=self.args.skip_dim,
                                     num_filters=self.args.num_filters[0], skip_size=self.args.skip_size,
                                     dropout_keep_prob=self.args.dropout_rate)

    def _setup_highway_layer(self):
        """
         Creating Highway Layer.
         """
        self.highway = HighwayLayer(in_units=self.args.fc_dim, out_units=self.args.fc_dim)

    def _setup_fc_layer(self):
        """
         Creating FC Layer.
         """
        # TODO
        # self.fc = nn.Linear(in_features=(self.args.rnn_dim * 2 + self.args.skip_size * self.args.skip_dim) * 3,
        #                     out_features=self.args.fc_dim, bias=True)
        self.fc = nn.Linear(in_features=self.args.rnn_dim, out_features=self.args.fc_dim, bias=True)
        self.out = nn.Linear(in_features=self.args.fc_dim, out_features=1, bias=True)

    def _setup_dropout(self):
        """
         Adding Dropout.
         """
        self.dropout = nn.Dropout(self.args.dropout_rate)

    def _setup_layers(self):
        """
        Creating layers of model.
        1. Embedding Layer.
        2. Convolution Layer.
        3. Bi-RNN Layer.
        4. Skip-RNN Layer.
        5. Highway Layer.
        6. FC Layer.
        7. Dropout
        """
        self._setup_embedding_layer()
        # self._setup_conv_layer()
        self._setup_bi_rnn_layer()
        # self._setup_skip_rnn_layer()
        self._setup_highway_layer()
        self._setup_fc_layer()
        self._setup_dropout()

    def forward(self, x, tsp):
        # Unpack the data first
        _pad, _len = rnn_utils.pad_packed_sequence(x, batch_first=True)

        embedded_sentence = self.embedding(_pad)
        # shape of embedded_sentence: [batch_size, batch_max_len, embedding_size]
        embedded_sentence = embedded_sentence.view(embedded_sentence.shape[0], embedded_sentence.shape[1], -1)

        # # Convolution Layer
        # # shape of conv_out: [batch_size, seq_len, num_filters[0]]
        # # shape of conv_avg: [batch_size, num_filters[0]]
        # conv_out_content, conv_avg_content = self.conv(embedded_sentence.permute(0, 2, 1))

        # Bi-RNN Layer
        # shape of rnn_out: [batch_size, rnn_hidden_size]
        rnn_out = self.bi_rnn(embedded_sentence, tsp)

        # # Skip-RNN Layer
        # # shape of skip_rnn_out: [batch_size, skip_size * skip_hidden_size]
        # if self.args.skip_size > 0:
        #     skip_rnn_out = self.skip_rnn(conv_out_content)
        #
        # # Concat Layer
        # combine_content = torch.cat((rnn_avg_content, skip_rnn_out_content), dim=1)
        #
        # # Concat
        # # shape of fc_in: [batch_size, (rnn_hidden_size + skip_size * skip_hidden_size) * 3]
        # # att_out = torch.cat((attention_cq, conv_avg_question, attention_oq), dim=1)
        # fc_in = torch.cat((combine_content), dim=1)
        #
        # Fully Connected Layer
        fc_out = self.fc(rnn_out)

        # Highway Layer
        highway_out = self.highway(fc_out)

        # Dropout
        h_drop = self.dropout(highway_out)

        logits = self.out(h_drop).squeeze()
        scores = torch.sigmoid(logits)

        return logits, scores


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.BCELoss = nn.BCELoss(reduce=True, size_average=True)

    def forward(self, predict_y, input_y):
        # Loss
        loss = self.BCELoss(predict_y, input_y)

        # value = (predict_y[0] - predict_y[1]) - (input_y[0] - input_y[1])
        # losses = torch.mean(torch.pow(value, 2))
        return loss
