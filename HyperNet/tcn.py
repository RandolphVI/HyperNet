# -*- coding:utf-8 -*-
__author__ = 'randolph'

"""Temporal Convolution Net."""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class TemporalBlock(nn.Module):
    def __init__(self, input_units, output_units, filter_size, stride_size, padding_size, dilation_size, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # Convolution Layer 1
        self.conv1 = weight_norm(nn.Conv1d(in_channels=input_units, out_channels=output_units, kernel_size=filter_size,
                                           stride=stride_size, padding=padding_size, dilation=dilation_size))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Convolution Layer 2
        self.conv2 = weight_norm(nn.Conv1d(in_channels=output_units, out_channels=output_units, kernel_size=filter_size,
                                           stride=stride_size, padding=padding_size, dilation=dilation_size))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                 self.conv2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(input_units, output_units, 1) if input_units != output_units else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, input_x):
        out = self.net(input_x)
        res = input_x if self.downsample is None else self.downsample(input_x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, input_units, num_filters, filter_size, padding_sizes, dilation_sizes, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_filters)
        for i in range(num_levels):
            dilation_size = dilation_sizes[i]
            padding_size = padding_sizes[i]
            in_channels = input_units if i == 0 else num_filters[i-1]
            out_channels = num_filters[i]
            layers += [TemporalBlock(in_channels, out_channels, filter_size, stride_size=1, padding_size=padding_size,
                                     dilation_size=dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, input_x):
        tcn_out = self.network(input_x)
        tcn_out = tcn_out.permute(0, 2, 1)
        tcn_avg = torch.mean(tcn_out, dim=1)
        return tcn_out, tcn_avg