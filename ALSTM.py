import torch
import torch.nn as nn


class ALSTMModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, output_size=10, num_layers=2, dropout=0.0, rnn_type="GRU"):
        super().__init__()
        self.name = 'ALSTM'
        self.hid_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.rnn_layer = num_layers
        self._build_model()
        self.name = 'ALSTM'

    def _build_model(self):
        try:
            klass = getattr(nn, self.rnn_type.upper())
        except Exception as e:
            raise ValueError("unknown rnn_type `%s`" % self.rnn_type) from e
        self.net = nn.Sequential()
        self.net.add_module("fc_in", nn.Linear(in_features=self.input_size, out_features=self.hid_size))
        self.net.add_module("act", nn.Tanh())
        self.rnn = klass(
            input_size=self.hid_size,
            hidden_size=self.hid_size,
            num_layers=self.rnn_layer,
            batch_first=True,
            dropout=self.dropout,
        )
        self.fc_out = nn.Linear(in_features=self.hid_size * 2, out_features=self.output_size)
        self.norm1 = nn.BatchNorm1d(self.input_size)
        if self.output_size != 1:
            self.norm2 = nn.BatchNorm1d(self.output_size)
        self.att_net = nn.Sequential()
        self.att_net.add_module(
            "att_fc_in",
            nn.Linear(in_features=self.hid_size, out_features=int(self.hid_size / 2)),
        )
        self.att_net.add_module("att_dropout", torch.nn.Dropout(self.dropout))
        self.att_net.add_module("att_act", nn.Tanh())
        self.att_net.add_module(
            "att_fc_out",
            nn.Linear(in_features=int(self.hid_size / 2), out_features=1, bias=False),
        )
        self.att_net.add_module("att_softmax", nn.Softmax(dim=1))

    def forward(self, inputs):
        # inputs: [batch_size, input_size*input_day]
        # inputs = inputs.view(len(inputs), self.input_size, -1)
        # inputs = inputs.permute(0, 2, 1)  # [batch, input_size, seq_len] -> [batch, seq_len, input_size]
        inputs = inputs.permute(0, 2, 1)
        norm_out = self.norm1(inputs)
        norm_out = norm_out.permute(0, 2, 1)
        rnn_out, _ = self.rnn(self.net(norm_out))  # [batch, seq_len, num_directions * hidden_size]
        attention_score = self.att_net(rnn_out)  # [batch, seq_len, 1]
        out_att = torch.mul(rnn_out, attention_score)
        out_att = torch.sum(out_att, dim=1)
        out = self.fc_out(
            torch.cat((rnn_out[:, -1, :], out_att), dim=1)
        )  # [batch, seq_len, num_directions * hidden_size] -> [batch, output_size]
        out = out.squeeze()
        if self.output_size != 1:
            out = self.norm2(out)
        return out