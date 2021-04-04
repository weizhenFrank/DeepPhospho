import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from .auxiliary_loss_transformer import TransformerEncoderLayer, TransformerEncoder
from .ion_model import PositionalEncoding


class LSTMTransformer(nn.Module):
    def __init__(self, RT_mode, ntoken, use_prosit=False, pdeep2mode=False, two_stage=False, model_name="LSTMTransformer", embed_dim=256,
                 lstm_hidden_dim=512, lstm_layers=2, lstm_num=2, bidirectional=True, hidden_dropout_prob=0.5,
                 max_len=100, num_attention_head=1, fix_lstm=False, pos_encode_dropout=0.1,
                 attention_dropout=0.1, num_encd_layer=1, transformer_hidden_dim=1024, ):

        super(LSTMTransformer, self).__init__()

        self.RT_mode = RT_mode

        if RT_mode:
            self.embedding = nn.Embedding(ntoken, embed_dim, padding_idx=0)
        else:
            if use_prosit:
                self.embedding = nn.Embedding(ntoken, embed_dim - 64, padding_idx=0)
                self.feature_embedding = nn.Linear(1, 32)
                self.feature_embedding2 = nn.Linear(1, 32)
            else:
                self.embedding = nn.Embedding(ntoken, embed_dim - 64, padding_idx=0)
                self.feature_embedding = nn.Linear(1, 64)

        self.lstm_list = nn.ModuleList()
        self.bidirectional = bidirectional
        self.use_prosit = use_prosit
        self.two_stage = two_stage
        if self.bidirectional:
            self.downdim_fcs = nn.ModuleList()

        in_channels = embed_dim

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

        self.layer_norm1 = LayerNorm(lstm_hidden_dim)
        self.layer_norm2 = LayerNorm(lstm_hidden_dim)

        self.pos_encoder = PositionalEncoding(lstm_hidden_dim, pos_encode_dropout, max_len=max_len)
        encoder_layers = TransformerEncoderLayer(lstm_hidden_dim, num_attention_head, transformer_hidden_dim,
                                                 attention_dropout=attention_dropout, hidden_dropout_prob=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encd_layer)
        self.outdim = lstm_hidden_dim

        if RT_mode:
            ion_len = 1
            # self.combine_steps = nn.Linear(1, 32)
        elif pdeep2mode:
            ion_len = 8
        elif use_prosit:
            ion_len = 6
        else:
            ion_len = 12

        if RT_mode:
            self.output_linear = nn.Sequential(
                nn.LeakyReLU(),
                nn.Linear(self.outdim, ion_len),
            )
            self.weight_layer = nn.Linear(self.outdim, 1)
            self.softmax = nn.Softmax(dim=1)
        else:
            self.output_linear = nn.Sequential(
                nn.LeakyReLU(),
                nn.Linear(self.outdim, ion_len),
                # nn.Sigmoid()
            )
        if two_stage:
            self.cls_linear = nn.Sequential(
                nn.LeakyReLU(),
                nn.Linear(self.outdim, ion_len),
            )
        self.transformer_flag = False

    def set_transformer(self, flag=True):
        self.transformer_flag = flag

    def forward(self, x1, x2=None, x3=None):

        if self.RT_mode:
            x = self.embedding(x1)

        else:
            if self.use_prosit:
                x1 = self.embedding(x1)  # (batch_sz, seq_len, feat_dim)
                x2 = self.feature_embedding(x2)  # (batch_sz, seq_len, feat_dim)
                x3 = self.feature_embedding2(x3)  # (batch_sz, seq_len, feat_dim)
                x = torch.cat((x1, x2, x3), dim=2)
            else:
                x1_embed = self.embedding(x1)  # (batch_sz, seq_len, feat_dim)
                x2_embed = self.feature_embedding(x2)  # (batch_sz, seq_len, feat_dim)
                x = torch.cat((x1_embed, x2_embed), dim=2)

        hidden_store = [x]
        for layer_i in range(len(self.lstm_list)):

            layer_hidden = self.lstm_list[layer_i](hidden_store[-1])[0]

            # (batch_sz, seq_len, feat_dim)

            if self.bidirectional:
                layer_hidden = self.downdim_fcs[layer_i](layer_hidden).clone()

            hidden_store.append(layer_hidden)

        hidden = hidden_store[-1]
        # hidden = x
        # hidden_store = []
        # for layer_i in range(len(self.lstm_list)):
        #
        #     hidden_store.append(self.lstm_list[layer_i](hidden)[0][:, :, :])
        #     hidden = hidden_store[-1]
        #     # (batch_sz, seq_len, feat_dim)
        #
        #     if self.bidirectional:
        #         hidden = self.downdim_fcs[layer_i](hidden)
        # if self.use_prosit:
        #     # todo maybe
        #     hidden = hidden[:, 1:, :]
        # else:
        #     hidden = hidden[:, :, :]

        if not self.transformer_flag:
            if self.RT_mode:
                hidden = hidden[:, 1, :]
                output = self.output_linear(hidden)
                return output
            else:
                if self.two_stage:
                    output = self.output_linear(hidden)
                    cls_out = self.cls_linear(hidden)
                    return output, cls_out
                else:
                    output = self.output_linear(hidden)
                    return output

            # ipdb.set_trace()

            # return custom_sigmoid(output)
        src = hidden.transpose(0, 1)  # (seq, bz, feat_dim)
        src = self.pos_encoder(src)
        src = self.layer_norm1(src)
        transformer_out, inter_out = self.transformer_encoder(src)
        transformer_out = transformer_out.transpose(0, 1)  # (bz, seq, feat_dim)
        # ipdb.set_trace()
        if self.use_prosit:
            transformer_out = transformer_out[:, 1:, :]
        # else:
        #     transformer_out = transformer_out[:, :, :]

        # if self.pretrain_mode:
        #     store = []
        #     output = self.layer_norm2(transformer_out)
        #     output = self.pretrain_decoder(output)
        #     store.append(output)
        # else:
        #     store = []
        layernormed_output = self.layer_norm2(transformer_out)
        # output = output[:, :, :]
        # store.append(output)

        # return custom_sigmoid(output)

        if self.two_stage:
            output = self.output_linear(layernormed_output)
            cls_out = self.cls_linear(layernormed_output)
            return output, cls_out
        else:
            if self.RT_mode:
                # padding_mask = src == 0
                # mean_mask = (padding_mask == 0).unsqueeze(dim=-1).float().cuda()

                # ipdb.set_trace()
                # output_layernorm = output_layernorm.sum(dim=1) / output_layernorm.shape[1]
                # ipdb.set_trace()
                raw_weight = self.weight_layer(layernormed_output)
                weight = self.softmax(raw_weight)
                trans = layernormed_output.transpose(1, 2)
                weighted_output = torch.matmul(trans, weight).squeeze()
                layernormed_output = weighted_output
                # output_layernorm = output_layernorm[:, 1, :]
            output = self.output_linear(layernormed_output)
            return output


# class Residual(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super(Residual, self).__init__()
#         self.conv1 = \
#             nn.Sequential(
#                 nn.Conv1d(in_channels=input_dim,
#                           out_channels=hidden_dim,
#                           kernel_size=3,
#                           stride=1,
#                           padding=int(3 / 2)),
#                 nn.BatchNorm1d(hidden_dim),
#                 nn.LeakyReLU(),
#             )
#         self.conv2 = \
#             nn.Sequential(
#                 nn.Conv1d(in_channels=hidden_dim,
#                           out_channels=hidden_dim,
#                           kernel_size=2,
#                           stride=1),
#                 nn.BatchNorm1d(hidden_dim),
#
#             )
#         self.conv3 = \
#             nn.Sequential(
#                 nn.Conv1d(in_channels=hidden_dim,
#                           out_channels=hidden_dim,
#                           kernel_size=2,
#                           padding=1,
#                           stride=1),
#                 nn.BatchNorm1d(hidden_dim),
#
#             )
#
#         self.conv4 = nn.Conv1d(in_channels=input_dim,
#                                out_channels=hidden_dim,
#                                kernel_size=1,
#                                stride=1,
#                                )
#     def forward(self, x):
#         y = self.conv1(x)
#         y = self.conv2(y)
#         y = self.conv3(y)
#         if self.conv4 is not None:
#             x = self.conv4(x)
#         return F.leaky_relu(x + y)

class Residual(nn.Module):
    def __init__(self, input_dim, hidden_dim, block_kernel_size=3, add_dropout=0.5, dilation=1):
        super(Residual, self).__init__()

        self.conv1 = \
            nn.Sequential(
                nn.Conv1d(in_channels=input_dim,
                          out_channels=hidden_dim,
                          kernel_size=block_kernel_size,
                          dilation=dilation,  # add dilation
                          stride=1,
                          padding=dilation * int(block_kernel_size / 2)),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(),
                # nn.Dropout(),  # add dropout trying to prevent overfitting
            )
        self.conv2 = \
            nn.Sequential(
                nn.Conv1d(in_channels=hidden_dim,
                          out_channels=hidden_dim,
                          kernel_size=block_kernel_size,
                          dilation=dilation,  # add dilation
                          stride=1,
                          padding=dilation * int(block_kernel_size / 2)),
                nn.BatchNorm1d(hidden_dim),

            )
        self.conv3 = nn.Conv1d(in_channels=input_dim,
                               out_channels=hidden_dim,
                               kernel_size=1,
                               stride=1,
                               )
        self.add_dropout = add_dropout

    def forward(self, x):
        y = F.dropout(self.conv1(x), p=self.add_dropout, training=self.training) if self.add_dropout else self.conv1(x)
        # add dropout trying to prevent overfit
        y = self.conv2(y)
        # ipdb.set_trace()
        if self.conv3 is not None:
            x = self.conv3(x)

        return F.dropout(F.leaky_relu(x + y), p=self.add_dropout, training=self.training) if self.add_dropout else F.leaky_relu(x + y)


class CNNTransformer(nn.Module):

    def __init__(self, ntoken, RT_mode, pdeep2mode=False, two_stage=False, model_name="CNNTransformer",
                 embed_dim=256, num_block=1, conv1_kernel_size=7, conv2_kernel_size=3, residual_dropout=0.5, dilation=1,
                 transformer_hidden_dim=512, num_attention_head=1, attention_dropout=0.1, num_encd_layer=2, dropout=0.1):

        super(CNNTransformer, self).__init__()
        self.RT_mode = RT_mode
        self.two_stage = two_stage
        if RT_mode:
            self.embedding = nn.Embedding(ntoken, embed_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(ntoken, embed_dim - 64, padding_idx=0)
            self.feature_embedding = nn.Linear(1, 64)

        transformer_in_dim = embed_dim * 2
        output_dim = transformer_in_dim
        self.outdim = output_dim

        self.text_cnn_conv = nn.Sequential(
            # nn.Conv2d(in_channels=1,
            #           out_channels=embed_dim,
            #           kernel_size=(embed_dim, conv1_kernel_size),
            #           stride=1,
            #           dilation=(1, dilation),
            #           padding=(0, dilation * int(conv1_kernel_size/2))),
            nn.Conv1d(in_channels=embed_dim,
                      out_channels=embed_dim,
                      kernel_size=conv1_kernel_size,
                      dilation=1,
                      stride=1,
                      padding=dilation * int(conv1_kernel_size / 2)),
            nn.LeakyReLU(),
        )

        self.cnn_feature = nn.Sequential(*[Residual(embed_dim, transformer_in_dim, conv2_kernel_size, residual_dropout, dilation)
                                           if i == 0
                                           else Residual(transformer_in_dim, transformer_in_dim, conv2_kernel_size, residual_dropout, dilation)
                                           for i in range(num_block)])

        self.layer_norm1 = LayerNorm(transformer_in_dim)
        self.layer_norm2 = LayerNorm(transformer_in_dim)

        self.pos_encoder = PositionalEncoding(transformer_in_dim)
        encoder_layers = TransformerEncoderLayer(transformer_in_dim, num_attention_head, transformer_hidden_dim,
                                                 attention_dropout=attention_dropout, hidden_dropout_prob=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encd_layer)

        if RT_mode:
            ion_len = 1
        elif pdeep2mode:
            ion_len = 8
        else:
            ion_len = 12

        if RT_mode:

            self.output_linear = nn.Sequential(
                nn.LeakyReLU(),
                nn.Linear(self.outdim, ion_len),
            )
            self.weight_layer = nn.Linear(self.outdim, 1)
            self.softmax = nn.Softmax(dim=1)
        else:
            self.output_linear = nn.Sequential(
                nn.LeakyReLU(),
                nn.Linear(self.outdim, ion_len),
                nn.Sigmoid()
            )
        if two_stage:
            self.cls_linear = nn.Sequential(
                nn.LeakyReLU(),
                nn.Linear(self.outdim, ion_len),
            )

    def forward(self, x1, x2=None):

        if self.RT_mode:
            x = self.embedding(x1)

        else:

            x1_embed = self.embedding(x1)  # (batch_sz, seq_len, feat_dim)
            x2_embed = self.feature_embedding(x2)  # (batch_sz, seq_len, feat_dim)
            x = torch.cat((x1_embed, x2_embed), dim=2)

        # x.transpose_(1, 2).unsqueeze_(1)  # (batch_sz, 1, feat_dim, seq_len) for 2D conv
        x.transpose_(1, 2)
        x = self.text_cnn_conv(x)
        # ipdb.set_trace()
        # x.squeeze_()
        x = self.cnn_feature(x)  # (batch_sz, feat_dim, seq_len)
        # ipdb.set_trace()

        if self.transformer_encoder is not None:
            x.transpose_(1, 2)  # (batch_sz, seq_len, feat_dim)
            x.transpose_(0, 1)  # (seq_len, batch_sz, feat_dim)
            src = self.pos_encoder(x)
            src = self.layer_norm1(src)
            transformer_out, inter_out = self.transformer_encoder(src)
            transformer_out = transformer_out.transpose(0, 1)  # (batch_sz, seq_len, feat_dim)

            layernormed_output = self.layer_norm2(transformer_out)

            if self.two_stage:
                output = self.output_linear(layernormed_output)
                cls_out = self.cls_linear(layernormed_output)
                return output, cls_out
            else:
                if self.RT_mode:
                    raw_weight = self.weight_layer(layernormed_output)
                    weight = self.softmax(raw_weight)
                    trans = layernormed_output.transpose(1, 2)
                    weighted_output = torch.matmul(trans, weight).squeeze()
                    layernormed_output = weighted_output

                output = self.output_linear(layernormed_output)
                return output


class CNN_LSTM_Transformer(nn.Module):
    def __init__(self, ntoken=31, embed_dim=256, lstm_hidden_dim=512, lstm_layers=2, lstm_num=2,
                 bidirectional=True, dropout=0.5, num_attention_head=16, transformer_hidden_dim=1024, num_encd_layer=32,
                 attention_dropout=0.1, pos_encode_dropout=0.1):

        super(CNN_LSTM_Transformer, self).__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(ntoken, embed_dim - 64, padding_idx=0)
        self.feature_embedding = nn.Linear(1, 64)
        # LSTM
        self.LeakyReLU = nn.LeakyReLU()
        self.lstm_list = nn.ModuleList()
        self.bidirectional = bidirectional

        if self.bidirectional:
            self.downdim_fcs = nn.ModuleList()
        for _ in range(lstm_num):
            if _ == 0:
                in_channels = embed_dim
            else:
                in_channels = lstm_hidden_dim

            self.lstm_list.append(
                nn.LSTM(input_size=in_channels,
                        hidden_size=lstm_hidden_dim,
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
                        nn.Linear(lstm_hidden_dim * 2, lstm_hidden_dim),
                    )
                )

        self.LSTM_outdim = lstm_hidden_dim
        # CNN
        self.text_cnn_conv = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=embed_dim,
                      kernel_size=(embed_dim, 5),
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
        self.CNN_outdim = embed_dim
        # Transformer

        self.layer_norm = LayerNorm(embed_dim)
        self.padding_idx = 0

        self.transformer_encoder = None
        self.pos_encoder = None
        if num_encd_layer is not None:
            self.pos_encoder = PositionalEncoding(embed_dim, pos_encode_dropout, max_len=100)
            encoder_layers = TransformerEncoderLayer(embed_dim, num_attention_head, transformer_hidden_dim,
                                                     attention_dropout=attention_dropout, hidden_dropout_prob=0.1)

            self.transformer_encoder = TransformerEncoder(encoder_layers, num_encd_layer)

        self.transformer_outdim = embed_dim
        self.init_weights()
        # ipdb.set_trace()

        self.outdim = self.LSTM_outdim + self.CNN_outdim + self.transformer_outdim
        self.final_output_linear = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(int(self.outdim), 256),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(256, 12)
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

    def forward(self, x1, x2):

        x1 = self.embedding(x1)  # (batch_sz, seq_len, feat_dim)
        x2 = self.feature_embedding(x2)  # (batch_sz, seq_len, feat_dim)
        x = torch.cat((x1, x2), dim=2)

        # LSTM
        hidden = x
        hidden_store = []
        for layer_i in range(len(self.lstm_list)):

            hidden_store.append(self.lstm_list[layer_i](hidden)[0][:, :, :])
            hidden = hidden_store[-1]
            # (batch_sz, seq_len, feat_dim)

            if self.bidirectional:
                hidden = self.downdim_fcs[layer_i](hidden)

            if layer_i == len(self.lstm_list) - 1:
                hidden_LSTM = hidden[:, :, :]
        # ipdb.set_trace()

        # CNN
        hidden = x.transpose(1, 2).unsqueeze(1)  # (batch_sz, 1, feat_dim, seq_len) for 2D conv
        # ipdb.set_trace()

        hidden = self.text_cnn_conv(hidden)
        # ipdb.set_trace()
        hidden = hidden.squeeze()
        hidden = self.cnn_feature(hidden)  # (batch_sz, feat_dim, seq_len)
        # ipdb.set_trace()
        hidden_CNN = hidden.transpose(2, 1)
        # ipdb.set_trace()

        # Transformer
        hidden = x * np.sqrt(self.embed_dim)
        src = hidden
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
        hidden_Transformer = transformer_out.transpose(0, 1)
        # ipdb.set_trace()

        hidden_total = torch.cat((hidden_LSTM, hidden_CNN, hidden_Transformer), dim=2)
        output = self.final_output_linear(hidden_total)
        return output


class LSTMTransformerEnsemble(nn.Module):
    def __init__(self, models: list, RT_mode: bool = False, two_stage: bool = False, model_name="LSTMTransformerEnsemble"):
        super(LSTMTransformerEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.RT_mode = RT_mode
        self.two_stage = two_stage
        self.transformer_flag = False

    def set_transformer(self, flag=True):
        for model in self.models:
            model.transformer_flag = flag

    def forward(self, x1, x2=None):
        if self.RT_mode:
            out = []
            for model in self.models:
                # ipdb.set_trace()
                out.append(model(x1=x1))
            # ipdb.set_trace()

            # out = torch.cat(out, dim=-1).mean(dim=-1)
            out = torch.cat(out, dim=-1)

            return out
        elif self.two_stage:
            out = []
            out_cls = []
            for model in self.models:
                out_per_model, out_cls_per_model = model(x1=x1, x2=x2)
                out.append(out_per_model)
                out_cls.append(out_cls_per_model)

            out = torch.cat(out, dim=-1).mean(dim=-1)
            out_cls = torch.cat(out_cls, dim=-1).mean(dim=-1)
            return out, out_cls
