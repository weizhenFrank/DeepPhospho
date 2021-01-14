import numpy as np
import copy

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer
from torch.nn import LayerNorm, Linear, Dropout
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import MultiheadAttention
from torch.nn.init import xavier_uniform_

from .transfromer_lib import _get_clones


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # Seq_len x 1 x feat_dim
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        # ipdb.set_trace()
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequnce to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for i in range(self.num_layers):
            output = self.layers[i](output, src_mask=mask,
                                    src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            output = self.norm(output)

        return output


class StackedLSTM(nn.Module):
    def __init__(self, ntoken, row_num, use_prosit, two_stage, model_name="StackedLSTM",
                 embed_dim=256, conv1_kernel=None, packed_sequence=False,
                 hidden_dim=512, lstm_layers=2, lstm_num=2, bidirectional=True,
                 dropout=0.5):
        super(StackedLSTM, self).__init__()

        self.packed_sequence = packed_sequence

        if use_prosit:
            self.embedding = nn.Embedding(ntoken, embed_dim - 64, padding_idx=0)
            self.feature_embedding = nn.Linear(1, 32)
            self.feature_embedding2 = nn.Linear(1, 32)
        else:
            self.embedding = nn.Embedding(ntoken, embed_dim - 64, padding_idx=0)
            self.feature_embedding = nn.Linear(1, 64)
        self.LeakyReLU = nn.LeakyReLU()
        self.lstm_list = nn.ModuleList()
        self.bidirectional = bidirectional
        self.use_prosit = use_prosit
        self.two_stage = two_stage

        if self.bidirectional:
            self.downdim_fcs = nn.ModuleList()
        for _ in range(lstm_num):
            if _ == 0:
                in_channels = embed_dim
            else:
                in_channels = hidden_dim

            self.lstm_list.append(
                nn.LSTM(input_size=in_channels,
                        hidden_size=hidden_dim,
                        num_layers=lstm_layers,
                        batch_first=True,
                        bidirectional=bidirectional,
                        dropout=dropout)
            )
            if self.bidirectional:
                self.downdim_fcs.append(
                    nn.Sequential(
                        nn.LeakyReLU(),
                        nn.Dropout(),
                        nn.Linear(hidden_dim * 2, hidden_dim),
                    )
                )

        output_dim = hidden_dim
        if two_stage:
            if use_prosit:
                self.output_linear = nn.Sequential(
                    nn.BatchNorm1d(row_num - 1),  # for prosit
                    # nn.BatchNorm1d(row_num)
                    nn.LeakyReLU(),
                    nn.Linear(int(output_dim), 6),
                )
                self.cls_layer = nn.Sequential(
                    nn.BatchNorm1d(row_num - 1),  # for prosit
                    nn.LeakyReLU(),
                    nn.Dropout(),
                    nn.Linear(int(output_dim), 6),
                )
            else:
                self.output_linear = nn.Sequential(
                    nn.BatchNorm1d(row_num),
                    nn.LeakyReLU(),
                    nn.Linear(int(output_dim), 12),

                )
                self.cls_layer = nn.Sequential(
                    nn.BatchNorm1d(row_num),
                    nn.LeakyReLU(),
                    nn.Dropout(),
                    nn.Linear(int(output_dim), 12),
                )

        else:
            if use_prosit:
                self.output_linear = nn.Sequential(
                    nn.BatchNorm1d(row_num - 1),  # for prosit
                    # nn.BatchNorm1d(row_num)
                    nn.LeakyReLU(),
                    nn.Linear(int(output_dim), 6),
                )
            else:
                self.output_linear = nn.Sequential(
                    nn.BatchNorm1d(row_num),
                    nn.LeakyReLU(),
                    nn.Linear(int(output_dim), 12),
                )

    def forward(self, x1, x2, x3=None):
        text_lengths = torch.sum(x1 != 0, dim=1)
        print(x1.shape)
        if self.use_prosit:
            x1 = self.embedding(x1)  # (batch_sz, seq_len, feat_dim)
            x2 = self.feature_embedding(x2)  # (batch_sz, seq_len, feat_dim)
            x3 = self.feature_embedding2(x3)  # (batch_sz, seq_len, feat_dim)
            x = torch.cat((x1, x2, x3), dim=2)
        else:
            x1 = self.embedding(x1)  # (batch_sz, seq_len, feat_dim)
            x2 = self.feature_embedding(x2)  # (batch_sz, seq_len, feat_dim)
            x = torch.cat((x1, x2), dim=2)
        hidden = x
        print(hidden.shape)
        hidden_store = []
        for layer_i in range(len(self.lstm_list)):

            # pack sequence manner is not recommended, padding tokens are important for LSTM models
            # to peceive the sequence length

            # packed_embedded = nn.utils.rnn.pack_padded_sequence(hidden, text_lengths,
            #                                                     enforce_sorted=False,
            #                                                     batch_first=True, )
            # packed_output = self.lstm_list[i](packed_embedded)[0]
            # # unpack sequence
            # hidden_store.append(nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, )[0][:])
            # hidden = hidden_store[-1]
            # feat_dim = int(hidden.shape[-1] / 2)
            # merged = hidden[:, :, :feat_dim] + hidden[:, :, -feat_dim:]
            # hidden = merged / 2  # (batch_sz, seq_len, feat_dim)
            # if i == len(self.lstm_list) - 1:
            #     # (batch_sz, seq_len, feat_dim)
            #     hidden = hidden[:, 0, :]
            # if layer_i == 0:
            #     hidden_store.append(hidden)
            hidden_store.append(self.lstm_list[layer_i](hidden)[0][:, :, :])
            hidden = hidden_store[-1]
            # (batch_sz, seq_len, feat_dim)

            if self.bidirectional:
                hidden = self.downdim_fcs[layer_i](hidden)

            # if layer_i == len(self.lstm_list) - 1:
            #     # (batch_sz, feat_dim)
            #     hidden = (hidden[:, -1, :] + hidden[:, 0, :])/2
            if layer_i == len(self.lstm_list) - 1:
                if self.use_prosit:
                    hidden = hidden[:, 1:, :]
                else:
                    hidden = hidden[:, :, :]

        # hidden.transpose_(1, 2)
        # hidden = self.global_avg(hidden).squeeze()

        bacth_norm = hidden.norm(dim=1)
        # ipdb.set_trace()
        output = self.output_linear(hidden)
        if self.two_stage:
            intensity_cls = self.cls_layer(hidden)

        # ipdb.set_trace()
        if self.two_stage:
            if self.training:
                return output, intensity_cls
            else:
                return output, bacth_norm
        else:
            if self.training:
                return output
            else:
                return output, bacth_norm


class LSTMTransformer(nn.Module):
    def __init__(self, ntoken, use_prosit, pdeep2mode, model_name="LSTMTransformer", embd_dim=256,
                 lstm_hidden_dim=512, lstm_layers=2, lstm_num=2, bidirectional=True, hidden_dropout_prob=0.5,
                 max_len=100, num_attention_head=1, fix_lstm=False, pos_encode_dropout=0.1,
                 attention_dropout=0.1, num_encd_layer=1, transformer_hidden_dim=1024, ):

        super(LSTMTransformer, self).__init__()

        if use_prosit:
            self.embedding = nn.Embedding(ntoken, embd_dim - 64, padding_idx=0)
            self.feature_embedding = nn.Linear(1, 32)
            self.feature_embedding2 = nn.Linear(1, 32)
        else:
            self.embedding = nn.Embedding(ntoken, embd_dim - 64, padding_idx=0)
            self.feature_embedding = nn.Linear(1, 64)
        self.embd_dim = embd_dim
        self.LeakyReLU = nn.LeakyReLU()
        self.lstm_list = nn.ModuleList()
        self.bidirectional = bidirectional
        self.use_prosit = use_prosit

        if self.bidirectional:
            self.downdim_fcs = nn.ModuleList()
        in_channels = embd_dim
        for _ in range(lstm_num):

            self.lstm_list.append(
                nn.LSTM(input_size=in_channels,
                        hidden_size=lstm_hidden_dim,
                        num_layers=lstm_layers,
                        batch_first=True,
                        bidirectional=True,
                        dropout=0.5)
            )
            in_channels = lstm_hidden_dim

            if self.bidirectional:
                self.downdim_fcs.append(
                    nn.Sequential(
                        nn.LeakyReLU(),
                        nn.Dropout(),
                        nn.Linear(lstm_hidden_dim * 2, lstm_hidden_dim),
                    )
                )

        self.layer_norm = LayerNorm(lstm_hidden_dim)
        self.pos_encoder = PositionalEncoding(lstm_hidden_dim, pos_encode_dropout, max_len=max_len)
        encoder_layers = TransformerEncoderLayer(lstm_hidden_dim, num_attention_head, transformer_hidden_dim,
                                                 dropout=attention_dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encd_layer)
        self.outdim = lstm_hidden_dim
        self.out_bn = nn.BatchNorm1d(self.outdim)
        if pdeep2mode:
            self.output_linear = nn.Sequential(
                nn.LeakyReLU(),
                nn.Linear(self.outdim, 8)
            )
        else:
            self.output_linear = nn.Sequential(
                nn.LeakyReLU(),
                nn.Linear(self.outdim, 12)
            )
        self.transformer_flag = False

    def set_transformer(self, flag=True):
        self.transformer_flag = flag

    def forward(self, x1, x2, x3=None):

        # print(x1.shape)
        if self.use_prosit:
            x1 = self.embedding(x1)  # (batch_sz, seq_len, feat_dim)
            x2 = self.feature_embedding(x2)  # (batch_sz, seq_len, feat_dim)
            x3 = self.feature_embedding2(x3)  # (batch_sz, seq_len, feat_dim)
            x = torch.cat((x1, x2, x3), dim=2)
        else:
            x1 = self.embedding(x1)  # (batch_sz, seq_len, feat_dim)
            x2 = self.feature_embedding(x2)  # (batch_sz, seq_len, feat_dim)
            x = torch.cat((x1, x2), dim=2)
        x = x * np.sqrt(self.embd_dim)
        hidden = x
        # print(hidden.shape)
        hidden_store = []
        for layer_i in range(len(self.lstm_list)):

            hidden_store.append(self.lstm_list[layer_i](hidden)[0][:, :, :])
            hidden = hidden_store[-1]
            # (batch_sz, seq_len, feat_dim)

            if self.bidirectional:
                hidden = self.downdim_fcs[layer_i](hidden)

            # if layer_i == len(self.lstm_list) - 1:
            #     # (batch_sz, feat_dim)
            #     hidden = (hidden[:, -1, :] + hidden[:, 0, :])/2
            if layer_i == len(self.lstm_list) - 1:
                if self.use_prosit:
                    hidden = hidden[:, 1:, :]
                else:
                    hidden = hidden[:, :, :]

        # hidden.transpose_(1, 2)
        # hidden = self.global_avg(hidden).squeeze()
        # hidden shape (bz, seq_length, hidden_dim) batchnorm1d expects (bz, channels) or (bz, channels, length)
        if not self.transformer_flag:
            hidden.transpose_(1, 2)
            # (bz, hidden_dim, seq_length)
            output = self.out_bn(hidden)
            output.transpose_(2, 1)
            return self.output_linear(output)

        src = hidden.transpose(0, 1)
        src = self.pos_encoder(src)
        src = self.layer_norm(src)
        hidden = self.transformer_encoder(src)

        transformer_out = hidden[:, :, :]
        # (seq_length, bz, hidden_dim)
        # ipdb.set_trace()
        # batchnorm1d expects (bz, channels) or (bz, channels, length)
        transformer_out.transpose_(0, 1).transpose_(1, 2)
        # ipdb.set_trace()

        # (seq_length, bz, hidden_dim) --> (bz, seq_length, hidden_dim) --> (bz,  hidden_dim, seq_length)
        transformer_out = self.out_bn(transformer_out)
        # ipdb.set_trace()

        transformer_out.transpose_(1, 2)
        # ipdb.set_trace()

        # (bz, hidden_dim, seq_length) --> (bz, seq_length, hidden_dim)
        output = self.output_linear(transformer_out)

        return output


class TransformerModel(nn.Module):

    def __init__(self, ntoken, use_prosit, pretrain_mode=False, max_len=100, model_name="Transformer",
                 embed_dim=256, num_attention_head=1,
                 hidden_dim=512, num_encd_layer=1,
                 pos_encode_dropout=0.1, attention_dropout=0.1, hidden_dropout_prob=0.1):

        """
        the transformer implementation using the pytorch lib

        - This transformer doesn't have intermediate layer which is the
         high dimension FC after the self attention
        - This transformer use the sin-cosine PositionalEncoding

        :param ntoken: the number of token, which set the dimension of one-hot vector
        :param embd_dim: the input dimension
        :param num_attention_head: the attention head number
        :param hidden_dim: the dimension of hidden layer(encoder)
        :param nun_encd_layer: the layer of encoder
        :param dropout:
        """
        super(TransformerModel, self).__init__()
        self.model_type = 'TransformerModel'
        assert model_name == self.model_type

        self.pretrain_mode = pretrain_mode
        # self.bn_all_input_feat = nn.BatchNorm1d(embed_dim)

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # self.embedding = PositionalEncodingEmbedding(ntoken, embd_dim)
        if pretrain_mode:
            self.embedding = nn.Embedding(ntoken, embed_dim, padding_idx=0)
        else:
            if use_prosit:
                self.embedding = nn.Embedding(ntoken, embed_dim - 64, padding_idx=0)
                self.feature_embedding = nn.Linear(1, 32)
                self.feature_embedding2 = nn.Linear(1, 32)
            else:
                self.embedding = nn.Embedding(ntoken, embed_dim - 64, padding_idx=0)
                self.feature_embedding = nn.Linear(1, 64)
        self.use_prosit = use_prosit
        self.layer_norm = LayerNorm(embed_dim)
        self.padding_idx = 0

        self.transformer_encoder = None
        self.pos_encoder = None
        if num_encd_layer is not None:
            self.pos_encoder = PositionalEncoding(embed_dim, pos_encode_dropout, max_len=max_len)
            encoder_layers = TransformerEncoderLayer(embed_dim, num_attention_head, hidden_dim,
                                                     dropout=attention_dropout, )

            self.transformer_encoder = TransformerEncoder(encoder_layers, num_encd_layer)

        self.init_weights()
        # ipdb.set_trace()

        self.outdim = embed_dim

        # self.pooler = nn.Sequential(
        #     # nn.BatchNorm1d(self.outdim),
        #     nn.Tanh(),
        #     nn.Dropout(),
        #     nn.Linear(self.outdim, self.outdim),
        #     # nn.Tanh()
        # )

        if use_prosit:
            self.output_linear = nn.Sequential(
                # nn.BatchNorm1d(row_num)
                nn.LeakyReLU(),
                nn.Dropout(),
                nn.Linear(self.outdim, 6),
            )
        else:
            self.output_linear = nn.Sequential(
                nn.LeakyReLU(),
                nn.Dropout(),
                nn.Linear(self.outdim, 12),
            )
        self.pretrain_decoder = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(self.outdim, ntoken),
        )

    def init_weights(self):
        def _init_weights(module):
            """ Initialize the weights
             this initialization is set by the HuggingFaceTransformer implementation
             this function is call by the nn.Module
             """
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        for each in self.modules():
            _init_weights(each)

        # if self.padding_idx is not None:
        #     with torch.no_grad():
        #         self.embedding.weight[self.padding_idx].fill_(0)

    def forward(self, x1, x2=None, x3=None):
        # print(x1.shape)
        if self.pretrain_mode:
            x = self.embedding(x1)
        else:
            if self.use_prosit:
                x1 = self.embedding(x1)  # (batch_sz, seq_len, feat_dim)
                x2 = self.feature_embedding(x2)  # (batch_sz, seq_len, feat_dim)
                x3 = self.feature_embedding2(x3)  # (batch_sz, seq_len, feat_dim)
                x = torch.cat((x1, x2, x3), dim=2)
            else:
                x1 = self.embedding(x1)  # (batch_sz, seq_len, feat_dim)
                x2 = self.feature_embedding(x2)  # (batch_sz, seq_len, feat_dim)
                x = torch.cat((x1, x2), dim=2)
        # x.transpose_(1, 2)
        # x = self.bn_all_input_feat(x)
        # x.transpose_(2, 1)
        x = x * np.sqrt(self.embed_dim)
        src = x

        src = src * np.sqrt(self.embed_dim)
        src = src.transpose(0, 1)  # (seq_length, bz, feat_dim)
        src = self.pos_encoder(src)
        src = self.layer_norm(src)
        hidden = self.transformer_encoder(src)

        transformer_out = hidden[:, :, :]
        # (seq_length, bz, hidden_dim)
        # ipdb.set_trace()
        # batchnorm1d expects (bz, channels) or (bz, channels, length)
        # transformer_out.transpose_(0, 1).transpose_(1, 2)
        # ipdb.set_trace()
        transformer_out.transpose_(0, 1)
        # (seq_length, bz, hidden_dim) --> (bz, seq_length, hidden_dim)
        # # (seq_length, bz, hidden_dim) --> (bz, seq_length, hidden_dim) --> (bz,  hidden_dim, seq_length)
        # transformer_out = self.out_bn(transformer_out)
        # ipdb.set_trace()

        # transformer_out.transpose_(1, 2)
        # ipdb.set_trace()

        # (bz, hidden_dim, seq_length) --> (bz, seq_length, hidden_dim)
        if self.use_prosit:
            transformer_out = transformer_out[:, 1:, :]
        else:
            transformer_out = transformer_out[:, :, :]
        if self.pretrain_mode:
            output = self.pretrain_decoder(transformer_out)
        else:
            output = self.output_linear(transformer_out)

        return output


class Residual(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Residual, self).__init__()
        self.conv1 = \
            nn.Sequential(
                nn.Conv1d(in_channels=input_dim,
                          out_channels=hidden_dim,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(),
            )
        self.conv2 = \
            nn.Sequential(
                nn.Conv1d(in_channels=hidden_dim,
                          out_channels=hidden_dim,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.BatchNorm1d(hidden_dim),

            )
        self.conv3 = nn.Conv1d(in_channels=input_dim,
                               out_channels=hidden_dim,
                               kernel_size=1,
                               stride=1,
                               )

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.conv3 is not None:
            x = self.conv3(x)
        return F.leaky_relu(x + y)


class CNNModel(nn.Module):
    def __init__(self, ntoken, use_prosit, model_name="CNNModel",
                 conv1_kernel_size=5, conv2_kernel_size=3,
                 embed_dim=256):

        super(CNNModel, self).__init__()
        if use_prosit:
            self.embedding = nn.Embedding(ntoken, embed_dim - 64, padding_idx=0)
            self.feature_embedding = nn.Linear(1, 32)
            self.feature_embedding2 = nn.Linear(1, 32)
        else:
            self.embedding = nn.Embedding(ntoken, embed_dim - 64, padding_idx=0)
            self.feature_embedding = nn.Linear(1, 64)

        self.use_prosit = use_prosit
        with torch.no_grad():
            self.embedding.weight[0].fill_(0)

        self.text_cnn_conv = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=embed_dim,
                      kernel_size=(embed_dim, conv1_kernel_size),
                      stride=1,
                      padding=(0, 2)),
            nn.LeakyReLU(),
        )

        self.cnn_feature = nn.Sequential(
            Residual(input_dim=embed_dim,
                     hidden_dim=embed_dim * 2),
            Residual(input_dim=embed_dim * 2,
                     hidden_dim=embed_dim * 4),
            Residual(input_dim=embed_dim * 4,
                     hidden_dim=embed_dim * 4),
            Residual(input_dim=embed_dim * 4,
                     hidden_dim=embed_dim * 4),
            Residual(input_dim=embed_dim * 4,
                     hidden_dim=embed_dim * 4),
            Residual(input_dim=embed_dim * 4,
                     hidden_dim=embed_dim)
        )
        output_dim = embed_dim

        if use_prosit:
            self.output_linear = nn.Sequential(
                # nn.BatchNorm1d(row_num)
                nn.LeakyReLU(),
                nn.Dropout(),
                nn.Linear(int(output_dim), 6),
            )
        else:
            self.output_linear = nn.Sequential(
                nn.LeakyReLU(),
                nn.Dropout(),
                nn.Linear(int(output_dim), 12),
            )

    def forward(self, x1, x2, x3=None):

        if self.use_prosit:
            x1 = self.embedding(x1)  # (batch_sz, seq_len, feat_dim)
            x2 = self.feature_embedding(x2)  # (batch_sz, seq_len, feat_dim)
            x3 = self.feature_embedding2(x3)  # (batch_sz, seq_len, feat_dim)
            x = torch.cat((x1, x2, x3), dim=2)
        else:
            x1 = self.embedding(x1)  # (batch_sz, seq_len, feat_dim)
            x2 = self.feature_embedding(x2)  # (batch_sz, seq_len, feat_dim)
            x = torch.cat((x1, x2), dim=2)

        # ipdb.set_trace()

        x.transpose_(1, 2).unsqueeze_(1)  # (batch_sz, 1, feat_dim, seq_len) for 2D conv
        # ipdb.set_trace()

        x = self.text_cnn_conv(x)
        # ipdb.set_trace()
        x.squeeze_()
        x = self.cnn_feature(x)  # (batch_sz, feat_dim, seq_len)

        hidden = x
        # ipdb.set_trace()
        bacth_norm = hidden.norm(dim=1)
        hidden.transpose_(2, 1)
        if self.use_prosit:
            hidden = hidden[:, 1:, :]
        else:
            hidden = hidden[:, :, :]
        output = self.output_linear(hidden)
        # ipdb.set_trace()

        if self.training:
            return output
        else:
            return output, bacth_norm


