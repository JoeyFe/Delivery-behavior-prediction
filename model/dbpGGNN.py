import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F


class GNN(Module):
    def __init__(self, hidden_size, step=3):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_gi = nn.Linear(self.input_size, self.gate_size, bias=True)
        self.linear_gh = nn.Linear(self.hidden_size, self.gate_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden))+ self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden))+ self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = self.linear_gi(inputs)
        gh = self.linear_gh(hidden)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SRGNN(nn.Module):
    def __init__(self, hidden_size, embedding_dim, att_hidden_size):
        super(SRGNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.att_hidden_size = att_hidden_size
        self.nonhybrid = False
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.attn = nn.Linear(self.hidden_size + self.hidden_size, self.att_hidden_size, bias=False)
        self.v = nn.Linear(self.att_hidden_size, 1, bias=False)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gnn = GNN(self.hidden_size, step=1)
        self.reset_parameters()
        self.emb = nn.Linear(53, hidden_size, bias=False)
        self.emb_two = nn.Linear(53, hidden_size, bias=False)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, enc_input, att_input, A, lengths):
        '''
            enc_input = [batch_size, N-action, n_feature]
            att_input = [batch_size, M-action, n_feature]
        '''
        # 初始化隐藏层长度
        # enc_input=self.emb(enc_input)
        hidden = self.gnn(A, enc_input)
        # 对填充后的数据进行处理
        ht = hidden[
            torch.arange(enc_input.shape[0]).long(), [seqlen - 1 for seqlen in lengths]]  # batch_size x latent_size

        q1 = self.linear_one(ht).unsqueeze(1)  # batch_size x 1 x latent_size
        N_action = enc_input.shape[1]
        q1_expand = q1.repeat(1, N_action, 1)

        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size

        alpha = self.v_t(torch.sigmoid(q1_expand + q2)).squeeze(2)
        c_local = torch.bmm(alpha.unsqueeze(1), hidden).squeeze(1)
        if not self.nonhybrid:
            c_t = self.linear_transform(torch.cat([c_local, ht], 1))
        else:
            c_t = c_local
        att_weight = torch.matmul(c_t.unsqueeze(1), att_input.transpose(1, 2)).squeeze(1)

        return F.softmax(att_weight, dim=1)
