import math
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import cosine_classsifier

SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # if use multi-GPU
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


class Model(nn.Module):
    """
    Model class
    """

    def __init__(self, input_dim, output_dim, config):
        super(Model, self).__init__()

        # Parameter
        self.hidden_dim = input_dim['hidden_dim']  # Chanel
        self.output_dim = output_dim
        self.feature_dim = input_dim['feature_dim']  # SeqLen
        self.dropout1 = nn.Dropout(config["trainer_dropout_feature"])
        self.dropout2 = nn.Dropout(config["trainer_dropout_hidden"])
        self.dtype = torch.float

        # Transformer
        # 定义 Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.hidden_dim,
                                                   dim_feedforward=self.hidden_dim * self.feature_dim,
                                                   dtype=self.dtype)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.embed = nn.Linear(self.hidden_dim, self.hidden_dim * self.feature_dim, bias=True, dtype=self.dtype)
        # 定义 Transformer 解码器
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_dim * self.feature_dim, nhead=self.feature_dim,
                                                   dim_feedforward=self.hidden_dim * self.feature_dim,
                                                   dtype=self.dtype)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        # Models
        self.Cnn = ConvNet(self.hidden_dim, [2, 1],
                           [self.hidden_dim * self.feature_dim, self.hidden_dim * self.feature_dim], [1, 1],
                           dtype=self.dtype)
        self.Rnn = RecNet(self.hidden_dim, self.hidden_dim * self.feature_dim, dtype=self.dtype)
        self.ResSeqLen = self.Cnn.output_seqlen(self.feature_dim) + self.feature_dim

        self.TF = Transformer(self.hidden_dim * self.feature_dim, self.hidden_dim * self.feature_dim, 2, 2,
                              config=config,
                              dtype=self.dtype)
        self.fc = nn.Linear(self.hidden_dim * self.feature_dim * self.ResSeqLen, self.hidden_dim, bias=True,
                            dtype=self.dtype)

        # FC classifier.
        self.output = nn.Linear(self.hidden_dim, self.output_dim, bias=False, dtype=self.dtype)
        self.bn = nn.BatchNorm1d(self.hidden_dim * self.feature_dim, dtype=self.dtype)
        self.relu = nn.ReLU()

        # cosine classifier
        # self.tau = nn.Parameter(torch.FloatTensor(1).fill_(10), requires_grad=True)
        # self.base_weights = nn.Parameter(torch.zeros(self.hidden_dim, self.output_dim), requires_grad=True)
        # nn.init.xavier_normal_(self.base_weights)

    def forward(self, x_env):
        """
        forward function.
        :param x: inputs
        :models:
        # # res_串
        # EmRes = (Embedding(src.permute(0, 2, 1))).permute(0, 2, 1)
        # PosRes = (Pos(EmRes.permute(2, 0, 1))).permute(1, 2, 0)
        # TFRes = ((TF(PosRes.permute(2, 0, 1), tgt.permute(2, 0, 1)))).permute(1, 2, 0)
        # CnnRes = Cnn(TFRes)
        # Cnn.output_seqlen(SeqLen)
        # PosCnnRes = (PosCnn(CnnRes.permute(2, 0, 1))).permute(1, 2, 0)
        # RnnRes = (Rnn(PosCnnRes.permute(0, 2, 1))).permute(0, 2, 1)
        # PosRnnRes = (PosRnn(RnnRes.permute(2, 0, 1))).permute(1, 2, 0)
        # 并
        # TFsrc=torch.concat((CnnRes,RnnRes), axis=2)
        # TFen=(TF.encode(TFsrc.permute(2, 0, 1))).permute(1, 2, 0)
        """
        # batch, feature, seq, cluster_dim, cyc
        x, env = x_env
        x = x[..., 1:]
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.concat((x, env), axis=1).to(self.dtype)
        # Transformer
        src_mask = nn.Transformer.generate_square_subsequent_mask(self.feature_dim, dtype=torch.bool)
        src_key_padding_mask = torch.zeros(x.shape[0], self.feature_dim, dtype=torch.bool)
        memory_key_padding_mask = torch.zeros(x.shape[0], self.feature_dim, dtype=torch.bool)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.ResSeqLen).type(torch.bool)
        tgt_key_padding_mask = torch.zeros(x.shape[0], self.ResSeqLen, dtype=torch.bool)
        TFencoded = (
            self.transformer_encoder(x.permute(2, 0, 1), mask=src_mask,
                                     src_key_padding_mask=src_key_padding_mask)).permute(1, 2, 0)

        TFencoded = (self.embed(TFencoded.permute(0, 2, 1))).permute(0, 2, 1)
        hidden = self.dropout1(x)
        # Models
        CnnRes = self.Cnn(hidden)
        RnnRes = (self.Rnn(hidden.permute(0, 2, 1))).permute(0, 2, 1)

        TFsrc = torch.concat((CnnRes, RnnRes), axis=2)

        TFdecoded = (self.transformer_decoder(TFsrc.permute(2, 0, 1), TFencoded.permute(2, 0, 1),
                                              tgt_mask=tgt_mask,
                                              tgt_key_padding_mask=tgt_key_padding_mask,
                                              memory_key_padding_mask=memory_key_padding_mask)).permute(1, 2, 0)

        TFsrc = torch.concat((TFsrc, TFdecoded), axis=2)
        tgtshape = torch.zeros(x.shape[0], self.hidden_dim * self.feature_dim,
                               self.hidden_dim * self.feature_dim * self.ResSeqLen, dtype=self.dtype)
        TFRes = (self.TF(TFsrc.permute(2, 0, 1), tgtshape.permute(2, 0, 1))).permute(1, 2, 0)

        # FC output
        feature = self.fc(TFRes)
        hidden = self.bn(feature)
        hidden = self.relu(hidden)
        hidden = self.dropout2(hidden)
        logits = self.output(hidden)
        logits = logits.reshape(logits.shape[0], -1)
        return F.softmax(logits, dim=1)

        # cosine output
        # logits = self.tau * cosine_classsifier(x=feature, w=self.base_weights)
        # if not self.training:
        #     return logits, feature
        # else:
        #     return F.softmax(logits, dim=1), feature


########## Net
# shape
# batch_size 表示批次的大小。
# sequence_length 表示序列的长度。
# in_channels 表示每个时间步的特征向量的维度。
class RecNet(nn.Module):
    def __init__(self, input_size, output_dim, dtype=torch.float16):
        super(RecNet, self).__init__()
        self.dtype = dtype
        self.output_dim = output_dim
        self.rnn = nn.RNN(input_size, output_dim, batch_first=True, dtype=self.dtype)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.output_dim, dtype=self.dtype)  # 初始化隐藏状态
        out, _ = self.rnn(x, h0)
        return out


# shape
# sequence_length 表示序列的长度（即时间步数）。
# batch_size 表示批次的大小。
# in_channels 表示每个时间步的特征向量的维度。
class Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, config,
                 dtype=torch.float16):
        super(Transformer, self).__init__()
        self.dtype = dtype
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, nhead, config['trainer_dropout_hidden'], dtype=self.dtype) for _ in
             range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, nhead, config['trainer_dropout_feature'], dtype=self.dtype) for _ in
             range(num_decoder_layers)])

    def encode(self, src):
        for layer in self.encoder_layers:
            src = layer(src)
        return src

    def decode(self, tgt, memory):
        for layer in self.decoder_layers:
            tgt = layer(tgt, memory)
        return tgt

    def forward(self, src, tgt):
        memory = self.encode(src)
        output = self.decode(tgt, memory)
        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, dtype=torch.float16):
        super(TransformerEncoderLayer, self).__init__()
        self.dtype = dtype
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, dtype=self.dtype)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model, dtype=self.dtype)

    def forward(self, src):
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout(src2)
        src = self.norm(src)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, dtype=torch.float16):
        super(TransformerDecoderLayer, self).__init__()
        self.dtype = dtype
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, dtype=self.dtype)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, dtype=self.dtype)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model, dtype=self.dtype)

    def forward(self, tgt, memory):
        tgt2, _ = self.self_attn(tgt, tgt, tgt)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        tgt2, _ = self.multihead_attn(tgt, memory, memory)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=100, dtype=torch.float16):
        super(PositionalEncoding, self).__init__()
        self.dtype = dtype
        pe = torch.zeros(max_len, dim, dtype=self.dtype)
        position = torch.arange(0, max_len, dtype=self.dtype).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=self.dtype) * (
                -np.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.pe = pe

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


# shape
# batch_size 表示批次的大小。
# in_channels 表示输入通道的数量。
# sequence_length 表示序列的长度。
class ConvNet(nn.Module):
    def __init__(self, input_dim, kernels, channels, strides, dtype=torch.float16):
        super(ConvNet, self).__init__()
        self.dtype = dtype
        sequential_list = []
        for i, (channel, kernel, stride) in enumerate(zip(channels, kernels, strides)):
            if i == 0:
                sequential_list.append(nn.Conv1d(in_channels=input_dim, out_channels=channel, kernel_size=kernel,
                                                 stride=stride, dtype=self.dtype))
            else:
                sequential_list.append(nn.Conv1d(in_channels=channels[i - 1], out_channels=channel, kernel_size=kernel,
                                                 stride=stride, dtype=self.dtype))
            sequential_list.append(nn.BatchNorm1d(channel, dtype=self.dtype))
            sequential_list.append(nn.ReLU())

        self.conv_layers = nn.Sequential(*sequential_list)

    def forward(self, x):
        return self.conv_layers(x)

    def output_seqlen(self, sequence_length):
        for module in self.conv_layers:
            if isinstance(module, nn.Conv1d):
                kernel_w = module.kernel_size[0]
                stride_w = module.stride[0]
                sequence_length = math.floor((sequence_length - kernel_w) / stride_w + 1)
        return sequence_length
