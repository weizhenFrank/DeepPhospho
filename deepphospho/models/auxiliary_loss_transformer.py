import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear, Dropout
from torch.nn import Module
from torch.nn import MultiheadAttention

from .transfromer_lib import _get_clones, _get_activation_fn
from .ion_model import PositionalEncoding
from deepphospho.utils.utils_functions import custom_sigmoid


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
        inter_out = []
        output = src

        for i in range(self.num_layers):

            output = self.layers[i](output, src_mask=mask,
                                    src_key_padding_mask=src_key_padding_mask)
            if self.training:
                inter_out.append(output)

        if self.norm:
            output = self.norm(output)

        return output, inter_out


class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, activation="relu",
                 attention_dropout=0.1, hidden_dropout_prob=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=attention_dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(hidden_dropout_prob)  # intermediate feature dropout
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(hidden_dropout_prob)  # attention concate FC dropout
        self.dropout2 = Dropout(hidden_dropout_prob)  # output dropout

        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src1 = self.norm1(src)
        src2 = self.self_attn(src1, src1, src1, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src3 = src + self.dropout1(src2)   # attention dropout
        src4 = self.norm2(src3)
        if hasattr(self, "activation"):
            src5 = self.linear2(self.dropout(self.activation(self.linear1(src4))))  # intermediate feature dropout
        else:  # for backward compatibility
            src5 = self.linear2(self.dropout(F.relu(self.linear1(src4))))
        src = src3 + self.dropout2(src5)  # output dropout
        return src


class TransformerModel(nn.Module):

    def __init__(self, ntoken, use_prosit, pretrain_mode=False, aux_loss=True, max_len=100, model_name="Transformer",
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
        self.aux_loss = aux_loss
        # self.bn_all_input_feat = nn.BatchNorm1d(embed_dim)

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

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
        self.num_encd_layer = num_encd_layer
        if num_encd_layer is not None:
            self.pos_encoder = PositionalEncoding(embed_dim, pos_encode_dropout, max_len=max_len)
            encoder_layers = TransformerEncoderLayer(embed_dim, num_attention_head, hidden_dim,
                                                     attention_dropout=attention_dropout, hidden_dropout_prob=hidden_dropout_prob)

            self.transformer_encoder = TransformerEncoder(encoder_layers, num_encd_layer)

        self.init_weights()
        # ipdb.set_trace()

        self.outdim = embed_dim

        if aux_loss:
            self.inter_out_list = nn.ModuleList()
            for i in range(num_encd_layer-1):
                self.inter_out_list.append(nn.Linear(self.outdim, 12))
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
            nn.Linear(self.outdim, 8),  # 8 is 1,2,3,4 with their modi
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
        hidden, inter_out = self.transformer_encoder(src)

        transformer_out = hidden[:, :, :]
        inter_out = inter_out[:-1]
        # (seq_length, bz, hidden_dim)
        # ipdb.set_trace()
        # batchnorm1d expects (bz, channels) or (bz, channels, length)
        # transformer_out.transpose_(0, 1).transpose_(1, 2)
        # ipdb.set_trace()
        transformer_out = transformer_out.transpose(0, 1)

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
            output = self.layer_norm(transformer_out)
            output = self.pretrain_decoder(output)
        else:
            output = self.layer_norm(transformer_out)
            output = self.output_linear(output)
        if self.aux_loss and self.training:
            inter_inten_out = []
            for inter_layer_hidden, layer_j in zip(inter_out, range(self.num_encd_layer-1)):
                inter_layer_hidden = inter_layer_hidden.transpose(0, 1)
                inter_inten_out.append(self.inter_out_list[layer_j](inter_layer_hidden))
            return custom_sigmoid(output), custom_sigmoid(inter_inten_out)
        else:
            return custom_sigmoid(output)






