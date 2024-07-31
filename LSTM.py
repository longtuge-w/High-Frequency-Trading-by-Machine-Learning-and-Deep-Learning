# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, output_size=10, num_layers=2, dropout=0.0):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.norm1 = nn.BatchNorm1d(input_size)
        if output_size != 1:
            self.norm2 = nn.BatchNorm1d(output_size)

        self.input_size = input_size
        self.output_size = output_size
        self.name = 'LSTM'

    def forward(self, x):
        # # x: [N, F*T]
        # x = x.reshape(len(x), self.input_size, -1)  # [N, F, T]
        # x = x.permute(0, 2, 1)  # [N, T, F]
        x = x.permute(0, 2, 1)
        out = self.norm1(x)
        out = out.permute(0, 2, 1)
        out, _ = self.rnn(out)
        out = out[:, -1, :]
        out = self.fc_out(out).squeeze()
        if self.output_size != 1:
            out = self.norm2(out)
        return out