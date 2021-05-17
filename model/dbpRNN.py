# -*- coding: utf-8 -*-
# code by Joey
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils
import torch.optim as optim
import torch.nn.functional as F

# Model
class Encoder(nn.Module):
    def __init__(self, enc_input_dim, enc_hid_dim):
        super().__init__()
        self.rnn = nn.GRU(enc_input_dim, enc_hid_dim)

    def forward(self, enc_input):
        '''
        enc_input = [batch_size, N-action, n_feature]
        '''
        # enc_input = [N-action, batch_size, n_feature]
        enc_input = enc_input.transpose(0, 1)

        # enc_hidden = [n_layers * num_directions=1, batch_size, enc_hid_dim]
        _, enc_hidden = self.rnn(enc_input)  # if h_0 is not give, it will be set 0 acquiescently

        return enc_hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, att_input_dim, att_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim + att_input_dim, att_hid_dim, bias=False)
        self.v = nn.Linear(att_hid_dim, 1, bias=False)

    def forward(self, enc_hidden, att_input):
        # s = [batch_size, dec_hid_dim]
        # enc_hidden = [num_layers(=1) * num_directions(=1), batch_size, enc_hid_dim]
        # att_input=[batch_size, M_action, m_feature]
        M_action = att_input.shape[1]

        # repeat enc_hidden  src_len times
        # s = [batch_size, M_action, enc_hid_dim]
        s = enc_hidden.repeat(M_action, 1, 1).transpose(0, 1)

        # energy = [batch_size, M_action, att_hid_dim]
        cat = torch.cat((s, att_input), dim=2)
        energy = torch.tanh(self.attn(torch.cat((s, att_input), dim=2)))

        # att_weight = [batch_size, M_action]
        att_weight = self.v(energy).squeeze(2)

        return F.softmax(att_weight, dim=1)


class SeqAndAttention(nn.Module):
    def __init__(self, encoder, attention):
        super().__init__()
        self.encoder = encoder
        self.attention = attention

    def forward(self, enc_input, att_input):
        # enc_input = [batch_size, N-action, n_feature]
        # att_input = [batch_size, M-action, m_feature]
        enc_hidden = self.encoder(enc_input)
        # att_weight_softmax = [batch_size, M_action]
        att_weight_softmax = self.attention(enc_hidden, att_input)
        # att_out = [batch_size, M_action]
        att_out = att_weight_softmax

        return att_out
