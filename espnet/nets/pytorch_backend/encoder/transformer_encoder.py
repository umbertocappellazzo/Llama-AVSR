#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""

import copy
import torch
from espnet.nets.pytorch_backend.nets_utils import rename_state_dict
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention, RelPositionMultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding, RelPositionalEncoding
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import PositionwiseFeedForward
from espnet.nets.pytorch_backend.transformer.repeat import repeat

class EncoderLayer(torch.nn.Module):
    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        dropout_rate,
        layerscale=False,
    ):
        """Construct an EncoderLayer object."""
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm_mha = LayerNorm(size)  # for the MHA module
        self.norm_ff = torch.nn.BatchNorm1d(size)  # for the FNN module
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.layerscale = layerscale
        if layerscale:
            self.gamma_ff = torch.nn.Parameter(0.1 * torch.ones((size,)), requires_grad=True)
            self.gamma_mha = torch.nn.Parameter(0.1 * torch.ones((size,)), requires_grad=True)

    def forward(self, x_input, mask, cache=None):
        """Compute encoded features.

        :param torch.Tensor x_input: encoded source features (batch, max_time_in, size)
        :param torch.Tensor mask: mask for x (batch, max_time_in)
        :param torch.Tensor cache: cache for x (batch, max_time_in - 1, size)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

        # multi-headed self-attention module
        residual = x
        x = self.norm_mha(x)

        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]

        if pos_emb is not None:
            x_att = self.self_attn(x_q, x, x, pos_emb, mask)
        else:
            x_att = self.self_attn(x_q, x, x, mask)

        if self.layerscale:
            x = residual + self.gamma_mha * self.dropout(x_att)
        else:
            x = residual + self.dropout(x_att)

        # feed Forward Module
        residual = x
        x = self.norm_ff(x.transpose(1, 2)).transpose(1, 2)
        if self.layerscale:
            x = residual + self.gamma_ff * self.dropout(self.feed_forward(x))
        else:
            x = residual + self.dropout(self.feed_forward(x))

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        if pos_emb is not None:
            return (x, pos_emb), mask
        else:
            return x, mask


def _pre_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    rename_state_dict(prefix + "input_layer.", prefix + "embed.", state_dict)
    rename_state_dict(prefix + "norm.", prefix + "after_norm.", state_dict)


class TransformerEncoder(torch.nn.Module):
    """Transformer encoder module.

    :param int idim: input dim
    :param int attention_dim: dimention of attention
    :param int attention_heads: the number of heads of multi head attention
    :param int linear_units: the number of units of position-wise feed forward
    :param int num_blocks: the number of decoder blocks
    :param float dropout_rate: dropout rate
    :param float attention_dropout_rate: dropout rate in attention
    :param float positional_dropout_rate: dropout rate after adding positional encoding
    :param bool normalize_before: whether to use layer_norm before the first block
    :param str positionwise_layer_type: linear of conv1d
    :param int positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
    :param bool zero_triu: whether to zero the upper triangular part of attention matrix
    """

    def __init__(
        self,
        attention_dim=768,
        attention_heads=12,
        linear_units=3072,
        num_blocks=12,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.1,
        normalize_before=True,
        layer_drop_rate=0.0,
        last_norm=True,
        layerscale=False,
        attn_layer_type="mha",
    ):
        """Construct an Encoder object."""
        super(TransformerEncoder, self).__init__()
        self._register_load_state_dict_pre_hook(_pre_hook)

        if attn_layer_type == "mha":
            pos_enc_class = PositionalEncoding
            encoder_attn_layer = MultiHeadedAttention
            encoder_attn_layer_args = (attention_heads, attention_dim, attention_dropout_rate)
        elif attn_layer_type == "rel_mha":
            pos_enc_class = RelPositionalEncoding
            encoder_attn_layer = RelPositionMultiHeadedAttention
            encoder_attn_layer_args = (attention_heads, attention_dim, attention_dropout_rate, False)


        self.embed = torch.nn.Sequential(pos_enc_class(attention_dim, positional_dropout_rate))
        self.normalize_before = normalize_before

        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (attention_dim, linear_units, dropout_rate)


        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                attention_dim,
                encoder_attn_layer(*encoder_attn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                layerscale=layerscale,
            ),
            layer_drop_rate=0.0,
        )
        self.after_norm = None
        if self.normalize_before and last_norm:
            self.after_norm = LayerNorm(attention_dim)

    def forward(self, xs, masks):
        """Encode input sequence.

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        xs = self.embed(xs)

        xs, masks = self.encoders(xs, masks)

        if isinstance(xs, tuple):
            xs = xs[0]

        if self.after_norm:
            xs = self.after_norm(xs)

        return xs, masks

    def forward_one_step(self, xs, masks, cache=None):
        """Encode input frame.

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :param List[torch.Tensor] cache: cache tensors
        :return: position embedded tensor, mask and new cache
        :rtype Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        xs, masks = self.embed(xs, masks)

        if cache is None:
            cache = [None for _ in range(len(self.encoders))]
        new_cache = []
        for c, e in zip(cache, self.encoders):
            xs, masks = e(xs, masks, cache=c)
            new_cache.append(xs)
        if self.after_norm:
            xs = self.after_norm(xs)
        return xs, masks, new_cache
