import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class NARM(nn.Module):
    def __init__(self, n_features, hidden_size, embedding_dim, att_hidden_size, n_layers=1):
        super(NARM, self).__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.att_hidden_size = att_hidden_size
        self.emb = nn.Embedding(self.n_features, self.embedding_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(0.25)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.n_layers)
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        # self.ct_dropout = nn.Dropout(0.5)
        self.b = nn.Linear(self.embedding_dim, 2 * self.hidden_size, bias=False)
        self.attn = nn.Linear(2 * self.hidden_size + self.hidden_size, self.att_hidden_size, bias=False)
        self.v = nn.Linear(self.att_hidden_size, 1, bias=False)
        # self.sf = nn.Softmax()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, enc_input, att_input, lengths):
        '''
            enc_input = [batch_size, N-action, n_feature]
            att_input = [batch_size, M-action, n_feature]
        '''
        # 初始化隐藏层长度
        # hidden = self.init_hidden(enc_input.size(0))
        # enc_input = [N-action, batch_size, n_feature]
        enc_input = enc_input.transpose(0, 1)
        # emb and dropout
        embs = enc_input
        # 对填充后的数据进行处理
        embs = pack_padded_sequence(embs, lengths)
        # gru_out = [N-action, batch_size, hid_dim]
        gru_out, hidden = self.gru(embs)
        gru_out, lengths = pad_packed_sequence(gru_out)
        # ht = [n_layers * num_directions=1, batch_size, enc_hid_dim]
        # fetch the last hidden state of last timestamp
        ht = hidden[-1]
        # 将tensor的维度换位 [batch_size, N-action, n_feature]
        gru_out = gru_out.permute(1, 0, 2)

        c_global = ht
        q1 = self.a_1(gru_out.contiguous().view(-1, self.hidden_size)).view(gru_out.size())
        q2 = self.a_2(ht).unsqueeze(1)

        mask = torch.where(enc_input == 0,
                           torch.tensor([0.], device=self.device), torch.tensor([1.], device=self.device)).transpose(0, 1)
        # q2_expand = q2.expand_as(q1)
        N_action = enc_input.shape[0]
        q2_expand = q2.repeat(1, N_action, 1)
        q2_masked = mask * q2_expand

        alpha = self.v_t(torch.sigmoid(q1 + q2_masked)).squeeze(2)
        c_local = torch.bmm(alpha.unsqueeze(1), gru_out).squeeze(1)

        # c_t [batch_size, n_feature*2]
        c_t = torch.cat([c_local, c_global], 1)
        c_t= self.linear_transform(c_t)
        att_weight = torch.matmul(c_t.unsqueeze(1), att_input.transpose(1, 2)).squeeze(1)

        return F.softmax(att_weight, dim=1)

