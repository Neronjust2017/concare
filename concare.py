import numpy as np
import argparse
import os
import imp
import re
import pickle
import datetime
import random
import math
import copy

RANDOM_SEED = 12345
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F

torch.manual_seed(RANDOM_SEED)  # cpu
torch.cuda.manual_seed(RANDOM_SEED)  # gpu
torch.backends.cudnn.deterministic = True  # cudnn

from utils import utils
from utils.readers import InHospitalMortalityReader
from utils.preprocessing import Discretizer, Normalizer
from utils import metrics
from utils import common_utils

### Prepare

data_path = 'data/'
file_name = 'model/concare'
small_part = False
arg_timestep = 1.0
batch_size = 256
epochs = 100

# Build readers, discretizers, normalizers
train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(data_path, 'train'),
                                         listfile=os.path.join(data_path, 'train_listfile.csv'),
                                         period_length=48.0)

val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(data_path, 'train'),
                                       listfile=os.path.join(data_path, 'val_listfile.csv'),
                                       period_length=48.0)

discretizer = Discretizer(timestep=arg_timestep,
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
normalizer_state = 'ihm_normalizer'
normalizer_state = os.path.join(os.path.dirname(data_path), normalizer_state)
normalizer.load_params(normalizer_state)

# %%

n_trained_chunks = 0
train_raw = utils.load_data(train_reader, discretizer, normalizer, small_part, return_names=True)
val_raw = utils.load_data(val_reader, discretizer, normalizer, small_part, return_names=True)

# %%

demographic_data = []
diagnosis_data = []
idx_list = []

demo_path = data_path + 'demographic/'
for cur_name in os.listdir(demo_path):
    cur_id, cur_episode = cur_name.split('_', 1)
    cur_episode = cur_episode[:-4]
    cur_file = demo_path + cur_name

    with open(cur_file, "r") as tsfile:
        header = tsfile.readline().strip().split(',')
        if header[0] != "Icustay":
            continue
        cur_data = tsfile.readline().strip().split(',')

    if len(cur_data) == 1:
        cur_demo = np.zeros(12)
        cur_diag = np.zeros(128)
    else:
        if cur_data[3] == '':
            cur_data[3] = 60.0
        if cur_data[4] == '':
            cur_data[4] = 160
        if cur_data[5] == '':
            cur_data[5] = 60

        cur_demo = np.zeros(12)
        cur_demo[int(cur_data[1])] = 1
        cur_demo[5 + int(cur_data[2])] = 1
        cur_demo[9:] = cur_data[3:6]
        cur_diag = np.array(cur_data[8:], dtype=np.int)

    demographic_data.append(cur_demo)
    diagnosis_data.append(cur_diag)
    idx_list.append(cur_id + '_' + cur_episode)

for each_idx in range(9, 12):
    cur_val = []
    for i in range(len(demographic_data)):
        cur_val.append(demographic_data[i][each_idx])
    cur_val = np.array(cur_val)
    _mean = np.mean(cur_val)
    _std = np.std(cur_val)
    _std = _std if _std > 1e-7 else 1e-7
    for i in range(len(demographic_data)):
        demographic_data[i][each_idx] = (demographic_data[i][each_idx] - _mean) / _std

# %%

device = torch.device("cuda:0" if torch.cuda.is_available() == True else 'cpu')
# device = torch.device('cpu')
print("available device: {}".format(device))

# %% md

### model

# %%

import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F

RANDOM_SEED = 12345
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True


class SingleAttention(nn.Module):
    def __init__(self, attention_input_dim, attention_hidden_dim, attention_type='add', demographic_dim=12,
                 time_aware=False, use_demographic=False):
        super(SingleAttention, self).__init__()

        self.attention_type = attention_type
        self.attention_hidden_dim = attention_hidden_dim
        self.attention_input_dim = attention_input_dim
        self.use_demographic = use_demographic
        self.demographic_dim = demographic_dim
        self.time_aware = time_aware

        self.attn = None

        # batch_time = torch.arange(0, batch_mask.size()[1], dtype=torch.float32).reshape(1, batch_mask.size()[1], 1)
        # batch_time = batch_time.repeat(batch_mask.size()[0], 1, 1)

        if attention_type == 'add':
            if self.time_aware == True:
                # self.Wx = nn.Parameter(torch.randn(attention_input_dim+1, attention_hidden_dim))
                self.Wx = nn.Parameter(torch.randn(attention_input_dim, attention_hidden_dim))
                self.Wtime_aware = nn.Parameter(torch.randn(1, attention_hidden_dim))
                nn.init.kaiming_uniform_(self.Wtime_aware, a=math.sqrt(5))
            else:
                self.Wx = nn.Parameter(torch.randn(attention_input_dim, attention_hidden_dim))
            self.Wt = nn.Parameter(torch.randn(attention_input_dim, attention_hidden_dim))
            self.Wd = nn.Parameter(torch.randn(demographic_dim, attention_hidden_dim))
            self.bh = nn.Parameter(torch.zeros(attention_hidden_dim, ))
            self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
            self.ba = nn.Parameter(torch.zeros(1, ))

            nn.init.kaiming_uniform_(self.Wd, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wx, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wt, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))
        elif attention_type == 'mul':
            self.Wa = nn.Parameter(torch.randn(attention_input_dim, attention_input_dim))
            self.ba = nn.Parameter(torch.zeros(1, ))

            nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))
        elif attention_type == 'concat':
            if self.time_aware == True:
                self.Wh = nn.Parameter(torch.randn(2 * attention_input_dim + 1, attention_hidden_dim))
            else:
                self.Wh = nn.Parameter(torch.randn(2 * attention_input_dim, attention_hidden_dim))

            self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
            self.ba = nn.Parameter(torch.zeros(1, ))

            nn.init.kaiming_uniform_(self.Wh, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))

        elif attention_type == 'new':
            self.Wt = nn.Parameter(torch.randn(attention_input_dim, attention_hidden_dim))
            self.Wx = nn.Parameter(torch.randn(attention_input_dim, attention_hidden_dim))

            self.rate = nn.Parameter(torch.ones(1))
            nn.init.kaiming_uniform_(self.Wx, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wt, a=math.sqrt(5))


        else:
            raise RuntimeError('Wrong attention type.')

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, demo=None):

        batch_size, time_step, input_dim = input.size()  # batch_size * time_step * hidden_dim(i)
        # assert(input_dim == self.input_dim)

        # time_decays = torch.zeros((time_step,time_step)).to(device)# t*t
        # for this_time in range(time_step):
        #     for pre_time in range(time_step):
        #         if pre_time > this_time:
        #             break
        #         time_decays[this_time][pre_time] = torch.tensor(this_time - pre_time, dtype=torch.float32).to(device)
        # b_time_decays = tile(time_decays, 0, batch_size).view(batch_size,time_step,time_step).unsqueeze(-1).to(device)# b t t 1

        time_decays = torch.tensor(range(time_step - 1, -1, -1), dtype=torch.float32).unsqueeze(-1).unsqueeze(0).to(
            device)  # 1*t*1
        b_time_decays = time_decays.repeat(batch_size, 1, 1)  # b t 1

        if self.attention_type == 'add':  # B*T*I  @ H*I
            q = torch.matmul(input[:, -1, :], self.Wt)  # b h
            q = torch.reshape(q, (batch_size, 1, self.attention_hidden_dim))  # B*1*H
            if self.time_aware == True:
                # k_input = torch.cat((input, time), dim=-1)
                k = torch.matmul(input, self.Wx)  # b t h
                # k = torch.reshape(k, (batch_size, 1, time_step, self.attention_hidden_dim)) #B*1*T*H
                time_hidden = torch.matmul(b_time_decays, self.Wtime_aware)  # b t h
            else:
                k = torch.matmul(input, self.Wx)  # b t h
                # k = torch.reshape(k, (batch_size, 1, time_step, self.attention_hidden_dim)) #B*1*T*H
            if self.use_demographic == True:
                d = torch.matmul(demo, self.Wd)  # B*H
                d = torch.reshape(d, (batch_size, 1, self.attention_hidden_dim))  # b 1 h
            h = q + k + self.bh  # b t h
            if self.time_aware == True:
                h += time_hidden
            h = self.tanh(h)  # B*T*H
            e = torch.matmul(h, self.Wa) + self.ba  # B*T*1
            e = torch.reshape(e, (batch_size, time_step))  # b t
        elif self.attention_type == 'mul':
            e = torch.matmul(input[:, -1, :], self.Wa)  # b i
            e = torch.matmul(e.unsqueeze(1), input.permute(0, 2, 1)).squeeze() + self.ba  # b t
        elif self.attention_type == 'concat':
            q = input[:, -1, :].unsqueeze(1).repeat(1, time_step, 1)  # b t i
            k = input
            c = torch.cat((q, k), dim=-1)  # B*T*2I
            if self.time_aware == True:
                c = torch.cat((c, b_time_decays), dim=-1)  # B*T*2I+1
            h = torch.matmul(c, self.Wh)
            h = self.tanh(h)
            e = torch.matmul(h, self.Wa) + self.ba  # B*T*1
            e = torch.reshape(e, (batch_size, time_step))  # b t

        elif self.attention_type == 'new':

            q = torch.matmul(input[:, -1, :], self.Wt)  # b h
            q = torch.reshape(q, (batch_size, 1, self.attention_hidden_dim))  # B*1*H
            k = torch.matmul(input, self.Wx)  # b t h
            dot_product = torch.matmul(q, k.transpose(1, 2)).squeeze()  # b t
            denominator = self.rate * torch.log(2.71828 + (1 - self.sigmoid(dot_product)) * (b_time_decays.squeeze()))
            e = self.tanh(dot_product / denominator)  # b * t# b * t

        # e = torch.exp(e - torch.max(e, dim=-1, keepdim=True).values)

        # if self.attention_width is not None:
        #     if self.history_only:
        #         lower = torch.arange(0, time_step).to(device) - (self.attention_width - 1)
        #     else:
        #         lower = torch.arange(0, time_step).to(device) - self.attention_width // 2
        #     lower = lower.unsqueeze(-1)
        #     upper = lower + self.attention_width
        #     indices = torch.arange(0, time_step).unsqueeze(0).to(device)
        #     e = e * (lower <= indices).float() * (indices < upper).float()

        # s = torch.sum(e, dim=-1, keepdim=True)
        # mask = subsequent_mask(time_step).to(device) # 1 t t 下三角
        # scores = e.masked_fill(mask == 0, -1e9)# b t t 下三角
        a = self.softmax(e)  # B*T
        self.attn = a
        v = torch.matmul(a.unsqueeze(1), input).squeeze()  # B*I

        return v, a


class FinalAttentionQKV(nn.Module):
    def __init__(self, attention_input_dim, attention_hidden_dim, attention_type='add', dropout=None):
        super(FinalAttentionQKV, self).__init__()

        self.attention_type = attention_type
        self.attention_hidden_dim = attention_hidden_dim
        self.attention_input_dim = attention_input_dim

        self.W_q = nn.Linear(attention_input_dim, attention_hidden_dim)
        self.W_k = nn.Linear(attention_input_dim, attention_hidden_dim)
        self.W_v = nn.Linear(attention_input_dim, attention_hidden_dim)

        self.W_out = nn.Linear(attention_hidden_dim, 1)

        self.b_in = nn.Parameter(torch.zeros(1, ))
        self.b_out = nn.Parameter(torch.zeros(1, ))

        nn.init.kaiming_uniform_(self.W_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_k.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_v.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_out.weight, a=math.sqrt(5))

        self.Wh = nn.Parameter(torch.randn(2 * attention_input_dim, attention_hidden_dim))
        self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
        self.ba = nn.Parameter(torch.zeros(1, ))

        nn.init.kaiming_uniform_(self.Wh, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))

        self.dropout = nn.Dropout(p=dropout)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):

        batch_size, time_step, input_dim = input.size()  # batch_size * input_dim + 1 * hidden_dim(i)
        input_q = self.W_q(input[:, -1, :])  # b h
        input_k = self.W_k(input)  # b t h
        input_v = self.W_v(input)  # b t h

        if self.attention_type == 'add':  # B*T*I  @ H*I

            q = torch.reshape(input_q, (batch_size, 1, self.attention_hidden_dim))  # B*1*H
            h = q + input_k + self.b_in  # b t h
            h = self.tanh(h)  # B*T*H
            e = self.W_out(h)  # b t 1
            e = torch.reshape(e, (batch_size, time_step))  # b t

        elif self.attention_type == 'mul':
            q = torch.reshape(input_q, (batch_size, self.attention_hidden_dim, 1))  # B*h 1
            e = torch.matmul(input_k, q).squeeze()  # b t

        elif self.attention_type == 'concat':
            q = input_q.unsqueeze(1).repeat(1, time_step, 1)  # b t h
            k = input_k
            c = torch.cat((q, k), dim=-1)  # B*T*2I
            h = torch.matmul(c, self.Wh)
            h = self.tanh(h)
            e = torch.matmul(h, self.Wa) + self.ba  # B*T*1
            e = torch.reshape(e, (batch_size, time_step))  # b t

        a = self.softmax(e)  # B*T
        if self.dropout is not None:
            a = self.dropout(a)
        v = torch.matmul(a.unsqueeze(1), input_v).squeeze()  # B*I

        return v, a


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(
        device)
    return torch.index_select(a, dim, order_index).to(device)


class PositionwiseFeedForward(nn.Module):  # new added

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x)))), None


class PositionalEncoding(nn.Module):  # new added / not use anymore

    def __init__(self, d_model, dropout, max_len=400):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0  # 下三角矩阵


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)  # b h t d_k
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)  # b h t t
    if mask is not None:  # 1 1 t t
        scores = scores.masked_fill(mask == 0, -1e9)  # b h t t 下三角
    p_attn = F.softmax(scores, dim=-1)  # b h t t
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn  # b h t v (d_k)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0):

        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, self.d_k * self.h), 3)
        self.final_linear = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)  # 1 1 t t

        nbatches = query.size(0)  # b
        input_dim = query.size(1)  # i+1
        feature_dim = query.size(-1)  # i+1

        # input size -> # batch_size * d_input * hidden_dim

        # d_model => h * d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]  # b num_head d_input d_k

        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)  # b num_head d_input d_v (d_k)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)  # batch_size * d_input * hidden_dim

        # DeCov
        DeCov_contexts = x.transpose(0, 1).transpose(1, 2)  # I+1 H B
        Covs = cov(DeCov_contexts[0, :, :])
        DeCov_loss = 0.5 * (torch.norm(Covs, p='fro') ** 2 - torch.norm(torch.diag(Covs)) ** 2)
        for i in range(feature_dim - 1 + 1):
            Covs = cov(DeCov_contexts[i + 1, :, :])
            DeCov_loss += 0.5 * (torch.norm(Covs, p='fro') ** 2 - torch.norm(torch.diag(Covs)) ** 2)

        return self.final_linear(x), DeCov_loss


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-7):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def cov(m, y=None):
    if y is not None:
        m = torch.cat((m, y), dim=0)
    m_exp = torch.mean(m, dim=1)
    x = m - m_exp[:, None]
    cov = 1 / (x.size(1) - 1) * x.mm(x.t())
    return cov


class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        returned_value = sublayer(self.norm(x))
        return x + self.dropout(returned_value[0]), returned_value[1]


class ConCare(nn.Module):
    def __init__(self, input_dim, hidden_dim, d_model, MHD_num_head, d_ff, output_dim, keep_prob=0.5):
        super(ConCare, self).__init__()

        # hyperparameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  # d_model
        self.d_model = d_model
        self.MHD_num_head = MHD_num_head
        self.d_ff = d_ff
        self.output_dim = output_dim
        self.keep_prob = keep_prob

        # layers
        self.PositionalEncoding = PositionalEncoding(self.d_model, dropout=0, max_len=400)

        self.GRUs = clones(nn.GRU(1, self.hidden_dim, batch_first=True), self.input_dim)
        self.LastStepAttentions = clones(
            SingleAttention(self.hidden_dim, 8, attention_type='concat', demographic_dim=12, time_aware=True,
                            use_demographic=False), self.input_dim)

        self.FinalAttentionQKV = FinalAttentionQKV(self.hidden_dim, self.hidden_dim, attention_type='mul',
                                                   dropout=1 - self.keep_prob)

        self.MultiHeadedAttention = MultiHeadedAttention(self.MHD_num_head, self.d_model, dropout=1 - self.keep_prob)
        self.SublayerConnection = SublayerConnection(self.d_model, dropout=1 - self.keep_prob)

        self.PositionwiseFeedForward = PositionwiseFeedForward(self.d_model, self.d_ff, dropout=0.1)

        self.demo_proj_main = nn.Linear(12, self.hidden_dim)
        self.demo_proj = nn.Linear(12, self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, self.output_dim)

        self.dropout = nn.Dropout(p=1 - self.keep_prob)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, input, demo_input):
        # input shape [batch_size, timestep, feature_dim]
        demo_main = self.tanh(self.demo_proj_main(demo_input)).unsqueeze(1)  # b hidden_dim

        batch_size = input.size(0)
        time_step = input.size(1)
        feature_dim = input.size(2)
        assert (feature_dim == self.input_dim)  # input Tensor : 256 * 48 * 76
        assert (self.d_model % self.MHD_num_head == 0)

        GRU_embeded_input = self.GRUs[0](input[:, :, 0].unsqueeze(-1),
                                         Variable(torch.zeros(batch_size, self.hidden_dim).unsqueeze(0)).to(device))[
            0]  # b t h
        Attention_embeded_input, self.gru_atten = self.LastStepAttentions[0](GRU_embeded_input)
        Attention_embeded_input = Attention_embeded_input.unsqueeze(1)  # b 1 h
        self.gru_atten = self.gru_atten.unsqueeze(1)  # b 1 h
        for i in range(feature_dim - 1):
            embeded_input = self.GRUs[i + 1](input[:, :, i + 1].unsqueeze(-1),
                                             Variable(torch.zeros(batch_size, self.hidden_dim).unsqueeze(0)).to(
                                                 device))[0]  # b 1 h
            embeded_input, atten = self.LastStepAttentions[i + 1](embeded_input)
            embeded_input = embeded_input.unsqueeze(1)  # b 1 h
            atten = atten.unsqueeze(1)  # b 1 h

            Attention_embeded_input = torch.cat((Attention_embeded_input, embeded_input), 1)  # b i h
            self.gru_atten = torch.cat((self.gru_atten, atten), 1)  # b i h

        Attention_embeded_input = torch.cat((Attention_embeded_input, demo_main), 1)  # b i+1 h
        posi_input = self.dropout(Attention_embeded_input)  # batch_size * d_input+1 * hidden_dim

        contexts = self.SublayerConnection(posi_input,
                                           lambda x: self.MultiHeadedAttention(posi_input, posi_input, posi_input,
                                                                               None))  # # batch_size * d_input * hidden_dim

        DeCov_loss = contexts[1]
        contexts = contexts[0]

        contexts = self.SublayerConnection(contexts, lambda x: self.PositionwiseFeedForward(contexts))[
            0]  # # batch_size * d_input * hidden_dim

        weighted_contexts = self.FinalAttentionQKV(contexts)[0]
        output = self.output(weighted_contexts)  # b 1
        output = self.sigmoid(output)

        return output, DeCov_loss


# %%

def get_loss(y_pred, y_true):
    loss = torch.nn.BCELoss()
    return loss(y_pred, y_true)


# %%

model = ConCare(input_dim=76, hidden_dim=32, d_model=32, MHD_num_head=4, d_ff=256, output_dim=1).to(device)
# input_dim, d_model, d_k, d_v, MHD_num_head, d_ff, output_dim
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# %%

class Dataset(data.Dataset):
    def __init__(self, x, y, name):
        self.x = x
        self.y = y
        self.name = name

    def __getitem__(self, index):  # 返回的是tensor
        return self.x[index], self.y[index], self.name[index]

    def __len__(self):
        return len(self.x)


# %%

train_dataset = Dataset(train_raw['data'][0], train_raw['data'][1], train_raw['names'])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataset = Dataset(val_raw['data'][0], val_raw['data'][1], val_raw['names'])
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# %% md

### Run

# %%

max_roc = 0
max_prc = 0
train_loss = []
train_model_loss = []
train_decov_loss = []
valid_loss = []
valid_model_loss = []
valid_decov_loss = []
history = []
np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

for each_epoch in range(epochs * 2):
    batch_loss = []
    model_batch_loss = []
    decov_batch_loss = []

    model.train()

    for step, (batch_x, batch_y, batch_name) in enumerate(train_loader):
        optimizer.zero_grad()
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)

        batch_demo = []
        for i in range(len(batch_name)):
            cur_id, cur_ep, _ = batch_name[i].split('_', 2)
            cur_idx = cur_id + '_' + cur_ep
            cur_demo = torch.tensor(demographic_data[idx_list.index(cur_idx)], dtype=torch.float32)
            batch_demo.append(cur_demo)

        batch_demo = torch.stack(batch_demo).to(device)
        output, decov_loss = model(batch_x, batch_demo)

        model_loss = get_loss(output, batch_y.unsqueeze(-1))
        loss = model_loss + 1000 * decov_loss

        batch_loss.append(loss.cpu().detach().numpy())
        model_batch_loss.append(model_loss.cpu().detach().numpy())
        decov_batch_loss.append(decov_loss.cpu().detach().numpy())
        loss.backward()
        optimizer.step()

        if step % 30 == 0:
            print('Epoch %d Batch %d: Train Loss = %.4f' % (each_epoch, step, np.mean(np.array(batch_loss))))
            print('Model Loss = %.4f, Decov Loss = %.4f' % (
            np.mean(np.array(model_batch_loss)), np.mean(np.array(decov_batch_loss))))
    train_loss.append(np.mean(np.array(batch_loss)))
    train_model_loss.append(np.mean(np.array(model_batch_loss)))
    train_decov_loss.append(np.mean(np.array(decov_batch_loss)))

    batch_loss = []
    model_batch_loss = []
    decov_batch_loss = []

    y_true = []
    y_pred = []
    with torch.no_grad():
        model.eval()
        for step, (batch_x, batch_y, batch_name) in enumerate(valid_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_demo = []
            for i in range(len(batch_name)):
                cur_id, cur_ep, _ = batch_name[i].split('_', 2)
                cur_idx = cur_id + '_' + cur_ep
                cur_demo = torch.tensor(demographic_data[idx_list.index(cur_idx)], dtype=torch.float32)
                batch_demo.append(cur_demo)

            batch_demo = torch.stack(batch_demo).to(device)
            output, decov_loss = model(batch_x, batch_demo)

            model_loss = get_loss(output, batch_y.unsqueeze(-1))

            loss = model_loss + 1000 * decov_loss
            batch_loss.append(loss.cpu().detach().numpy())
            model_batch_loss.append(model_loss.cpu().detach().numpy())
            decov_batch_loss.append(decov_loss.cpu().detach().numpy())
            y_pred += list(output.cpu().detach().numpy().flatten())
            y_true += list(batch_y.cpu().numpy().flatten())

    valid_loss.append(np.mean(np.array(batch_loss)))
    valid_model_loss.append(np.mean(np.array(model_batch_loss)))
    valid_decov_loss.append(np.mean(np.array(decov_batch_loss)))

    print("\n==>Predicting on validation")
    print('Valid Loss = %.4f' % (valid_loss[-1]))
    print('valid_model Loss = %.4f' % (valid_model_loss[-1]))
    print('valid_decov Loss = %.4f' % (valid_decov_loss[-1]))
    y_pred = np.array(y_pred)
    y_pred = np.stack([1 - y_pred, y_pred], axis=1)
    ret = metrics.print_metrics_binary(y_true, y_pred)
    history.append(ret)
    print()

    cur_auroc = ret['auroc']

    if cur_auroc > max_roc:
        max_roc = cur_auroc
        state = {
            'net': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': each_epoch
        }
        torch.save(state, file_name)
        print('\n------------ Save best model ------------\n')

# %% md

### Run for test

# %%

checkpoint = torch.load(file_name)
save_epoch = checkpoint['epoch']
model.load_state_dict(checkpoint['net'])
optimizer.load_state_dict(checkpoint['optimizer'])
model.eval()

test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(data_path, 'test'),
                                        listfile=os.path.join(data_path, 'test_listfile.csv'),
                                        period_length=48.0)
test_raw = utils.load_data(test_reader, discretizer, normalizer, small_part, return_names=True)
test_dataset = Dataset(test_raw['data'][0], test_raw['data'][1], test_raw['names'])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %%

batch_loss = []
y_true = []
y_pred = []
with torch.no_grad():
    model.eval()
    for step, (batch_x, batch_y, batch_name) in enumerate(test_loader):
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_demo = []
        for i in range(len(batch_name)):
            cur_id, cur_ep, _ = batch_name[i].split('_', 2)
            cur_idx = cur_id + '_' + cur_ep
            cur_demo = torch.tensor(demographic_data[idx_list.index(cur_idx)], dtype=torch.float32)
            batch_demo.append(cur_demo)

        batch_demo = torch.stack(batch_demo).to(device)
        output = model(batch_x, batch_demo)[0]

        loss = get_loss(output, batch_y.unsqueeze(-1))
        batch_loss.append(loss.cpu().detach().numpy())
        y_pred += list(output.cpu().detach().numpy().flatten())
        y_true += list(batch_y.cpu().numpy().flatten())

print("\n==>Predicting on test")
print('Test Loss = %.4f' % (np.mean(np.array(batch_loss))))
y_pred = np.array(y_pred)
y_pred = np.stack([1 - y_pred, y_pred], axis=1)
test_res = metrics.print_metrics_binary(y_true, y_pred)

