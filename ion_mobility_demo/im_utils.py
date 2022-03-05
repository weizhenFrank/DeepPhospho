_SEED = 0

import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(-1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_phospho.model_dataset.preprocess_input_data import ENDING_CHAR
from deep_phospho.models.auxiliary_loss_transformer import TransformerEncoderLayer, TransformerEncoder
from deep_phospho.models.ion_model import PositionalEncoding

random.seed(_SEED)
np.random.seed(_SEED)
torch.manual_seed(_SEED)
torch.cuda.manual_seed_all(_SEED)
torch.backends.cudnn.deterministic = True
torch.autograd.set_detect_anomaly(True)


class IMDataset(torch.utils.data.Dataset):
    def __init__(self, im_df, dictionary, logger, scale_range=None, name=None, has_im=True):
        self.prec_num = len(im_df)
        intpeps = im_df['IntPep'].tolist()
        charges = im_df['PrecCharge'].tolist()

        self.x_pep = np.zeros((self.prec_num, 52 + 2))
        self.x_charge = np.zeros((self.prec_num, 52 + 2))

        logger.info(f'Init dataset {name}, with total {self.prec_num} precursors')
        with tqdm(range(self.prec_num)) as t:
            for i in t:
                pep = intpeps[i] + ENDING_CHAR
                _pep_len = len(pep)
                c = charges[i]

                self.x_pep[i, :_pep_len] = [dictionary.word2idx[aa] for aa in pep]
                self.x_charge[i, :_pep_len] = [c for aa in pep]

        if has_im:
            self.y = im_df['IM'].values
            if scale_range is not None:
                self.y = (self.y - scale_range[0]) / (scale_range[1] - scale_range[0])
        else:
            self.y = np.zeros(self.prec_num)

        self.x_pep = torch.from_numpy(self.x_pep).long()
        self.x_charge = torch.from_numpy(self.x_charge).float()
        self.y = torch.from_numpy(self.y).float()

    def __getitem__(self, idx):
        return (self.x_pep[idx], self.x_charge[idx]), self.y[idx]

    def __len__(self):
        return self.prec_num


class LSTMTransformerForIM(nn.Module):
    def __init__(
            self,
            ntoken,
            embed_dim=256,
            lstm_hidden_dim=512,
            lstm_layers=2,
            lstm_num=2,
            bidirectional=True,
            max_len=100,
            num_attention_head=8,
            pos_encode_dropout=0.1,
            attention_dropout=0.1,
            num_encd_layer=8,
            transformer_hidden_dim=1024,
    ):
        super(LSTMTransformerForIM, self).__init__()

        self.embedding = nn.Embedding(ntoken, embed_dim - 64, padding_idx=0)
        self.feature_embedding = nn.Linear(1, 64)

        self.lstm_list = nn.ModuleList()
        self.downdim_fcs = nn.ModuleList()

        in_channels = embed_dim
        for _ in range(lstm_num):
            self.lstm_list.append(
                nn.LSTM(input_size=in_channels,
                        hidden_size=lstm_hidden_dim,
                        num_layers=lstm_layers,
                        batch_first=True,
                        bidirectional=bidirectional,
                        dropout=0.5)
            )
            in_channels = lstm_hidden_dim

            self.downdim_fcs.append(
                nn.Sequential(
                    nn.LeakyReLU(),
                    nn.Dropout(),
                    nn.Linear(lstm_hidden_dim * 2, lstm_hidden_dim),
                )
            )

        self.layer_norm1 = LayerNorm(lstm_hidden_dim)
        self.layer_norm2 = LayerNorm(lstm_hidden_dim)

        self.pos_encoder = PositionalEncoding(lstm_hidden_dim, pos_encode_dropout, max_len=max_len)
        encoder_layers = TransformerEncoderLayer(lstm_hidden_dim, num_attention_head, transformer_hidden_dim,
                                                 attention_dropout=attention_dropout, hidden_dropout_prob=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encd_layer)
        self.outdim = lstm_hidden_dim

        self.output_linear = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(self.outdim, 1),
        )
        self.weight_layer = nn.Linear(self.outdim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        x1_embed = self.embedding(x1)  # (batch_sz, seq_len, feat_dim)
        x2_embed = self.feature_embedding(x2.unsqueeze(-1))  # (batch_sz, seq_len, feat_dim)
        x = torch.cat((x1_embed, x2_embed), dim=2)

        hidden_store = [x]
        for layer_i in range(len(self.lstm_list)):
            layer_hidden = self.lstm_list[layer_i](hidden_store[-1])[0]
            layer_hidden = self.downdim_fcs[layer_i](layer_hidden).clone()
            hidden_store.append(layer_hidden)
        hidden = hidden_store[-1]

        src = hidden.transpose(0, 1)  # (seq, bz, feat_dim)
        src = self.pos_encoder(src)
        src = self.layer_norm1(src)
        transformer_out, inter_out = self.transformer_encoder(src)
        transformer_out = transformer_out.transpose(0, 1)  # (bz, seq, feat_dim)
        layernormed_output = self.layer_norm2(transformer_out)

        raw_weight = self.weight_layer(layernormed_output)
        weight = self.softmax(raw_weight)
        trans = layernormed_output.transpose(1, 2)
        weighted_output = torch.matmul(trans, weight).squeeze()
        layernormed_output = weighted_output
        output = self.output_linear(layernormed_output)
        return output


class CNN(nn.Module):
    def __init__(self, ntoken):
        super(CNN, self).__init__()

        self.embedding = nn.Embedding(ntoken, 256 - 64, padding_idx=0)
        self.feature_embedding = nn.Linear(1, 64)

        self.conv1 = nn.Sequential(
            nn.Conv1d(54, 128, 7, 4, 4),
            nn.MaxPool2d(2),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 256, 7, 4, 4),
            nn.MaxPool2d(2),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, 1, 1, 1),
            nn.MaxPool2d(2),
            nn.LeakyReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(192, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 1),
            nn.Softmax()
        )

    def forward(self, x1, x2):
        x1_embed = self.embedding(x1)
        x2_embed = self.feature_embedding(x2.unsqueeze(-1))
        x = torch.cat((x1_embed, x2_embed), dim=2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
