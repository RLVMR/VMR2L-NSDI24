"""
Implementation of the transformer model from Attention is All You Need.
References:
1. jadore801120/attention-is-all-you-need-pytorch
2. pytorch/pytorch
"""

import copy
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.container import ModuleList


class TransformerEncoder(Module):

    def __init__(self, encoder_layer, params):
        super(TransformerEncoder, self).__init__()
        self.transformer_blocks = params.transformer_blocks
        self.layers = _get_clones(encoder_layer, self.transformer_blocks)
        self.dropout = nn.Dropout(params.dropout)
        self.layer_norm = LayerNorm(params.transformer_input_size, eps=1e-6)

    def forward(self, output, return_attns=False, enc_position=None):

        output = self.layer_norm(output)
        output = self.dropout(output)

        if return_attns is False:
            # record position bias from first layer and share it with other layers
            output, _, enc_slf_bias = self.layers[0](src=output, return_position_bias=True,
                                                     enc_position=enc_position)
            for i in range(1, self.transformer_blocks):
                output, _ = self.layers[i](src=output, return_position_bias=False,
                                           enc_slf_bias=enc_slf_bias)
            output = self.layer_norm(output)
            output = self.dropout(output)
            return output
        else:  # return_attns is True
            enc_slf_attn_list = []
            # record position bias from first layer and share it with other layers
            output, enc_slf_attn, enc_slf_bias = self.layers[0](src=output, return_position_bias=True,
                                                                enc_position=enc_position)
            for i in range(1, self.transformer_blocks):
                output, enc_slf_attn = self.layers[i](src=output, return_position_bias=False,
                                                      enc_slf_bias=enc_slf_bias)
                enc_slf_attn_list += [enc_slf_attn]
            output = self.layer_norm(output)
            output = self.dropout(output)
            return output, enc_slf_attn_list[-1]


class TransformerDecoder(Module):

    def __init__(self, decoder_layer, params):
        super(TransformerDecoder, self).__init__()
        self.transformer_blocks = params.transformer_blocks
        self.layers = _get_clones(decoder_layer, self.transformer_blocks)
        self.dropout = nn.Dropout(params.dropout)
        self.layer_norm = LayerNorm(params.transformer_input_size, eps=1e-6)

    def forward(self, output, memory, return_attns=False, enc_position=None, dec_position=None):

        output = self.layer_norm(output)
        output = self.dropout(output)

        if return_attns is False:
            # record position bias from first layer and share it with other layers
            output, _, dec_slf_bias, enc_dec_bias = self.layers[0](tgt=output, memory=memory,
                                                                   return_position_bias=True,
                                                                   enc_position=enc_position,
                                                                   dec_position=dec_position)
            for i in range(1, self.transformer_blocks):
                output, _ = self.layers[i](tgt=output, memory=memory, return_position_bias=False,
                                           dec_slf_bias=dec_slf_bias,
                                           enc_dec_bias=enc_dec_bias)
            output = self.layer_norm(output)
            output = self.dropout(output)
            return output
        else:  # return_attns is True
            enc_dec_attn_list = []
            # record position bias from first layer and share it with other layers
            output, enc_dec_attn, dec_slf_bias, enc_dec_bias = self.layers[0](tgt=output,
                                                                              memory=memory,
                                                                              return_position_bias=True,
                                                                              enc_position=enc_position,
                                                                              dec_position=dec_position)
            enc_dec_attn_list += [enc_dec_attn]
            for i in range(1, self.transformer_blocks):
                output, enc_dec_attn = self.layers[i](tgt=output, memory=memory, return_position_bias=False,
                                                      dec_slf_bias=dec_slf_bias,
                                                      enc_dec_bias=enc_dec_bias)
                enc_dec_attn_list += [enc_dec_attn]
            output = self.layer_norm(output)
            output = self.dropout(output)
            return output, enc_dec_attn_list[-1]


class TransformerEncoderLayer(Module):

    def __init__(self, params):
        super(TransformerEncoderLayer, self).__init__()
        d_model = params.transformer_input_size
        self.slf_attn = MultiHeadAttention(params.num_head, d_model, params.d_kv, dropout=params.dropout,
                                           has_relative_attention_bias=params.has_relative_attention_bias,
                                           relative_attention_num_buckets=params.relative_attention_num_buckets)
        self.pos_ffn = PositionwiseFeedForward(d_model, params.d_ff, dropout=params.dropout)

    def forward(self, src, return_position_bias=False, enc_slf_bias=None, enc_position=None):
        if return_position_bias:
            enc_output, enc_slf_attn, enc_slf_bias = self.slf_attn(q_hid=src, return_position_bias=True,
                                                                   context_position=enc_position,
                                                                   memory_position=enc_position)
            enc_output = self.pos_ffn(enc_output)
            return enc_output, enc_slf_attn, enc_slf_bias
        else:
            enc_output, enc_slf_attn = self.slf_attn(q_hid=src, return_position_bias=False,
                                                     position_bias=enc_slf_bias)
            enc_output = self.pos_ffn(enc_output)
            return enc_output, enc_slf_attn


class TransformerDecoderLayer(Module):

    def __init__(self, params):
        super(TransformerDecoderLayer, self).__init__()
        d_model = params.transformer_input_size
        self.slf_attn = MultiHeadAttention(params.num_head, d_model, params.d_kv, dropout=params.dropout,
                                           has_relative_attention_bias=params.has_relative_attention_bias,
                                           relative_attention_num_buckets=params.relative_attention_num_buckets)
        self.enc_attn = MultiHeadAttention(params.num_head, d_model, params.d_kv, dropout=params.dropout,
                                           has_relative_attention_bias=params.has_relative_attention_bias,
                                           relative_attention_num_buckets=params.relative_attention_num_buckets)
        self.pos_ffn = PositionwiseFeedForward(d_model, params.d_ff, dropout=params.dropout)

    def forward(self, tgt, memory, return_position_bias=False, dec_slf_bias=None, enc_dec_bias=None,
                enc_position=None, dec_position=None):
        if return_position_bias:
            tgt, _, dec_slf_bias = self.slf_attn(q_hid=tgt, return_position_bias=True,
                                                 context_position=dec_position,
                                                 memory_position=dec_position)
            enc_output, enc_dec_attn, enc_dec_bias = self.enc_attn(q_hid=tgt, k_hid=memory, v_hid=memory,
                                                                   return_position_bias=True,
                                                                   context_position=dec_position,
                                                                   memory_position=enc_position)
            enc_output = self.pos_ffn(enc_output)
            return enc_output, enc_dec_attn, dec_slf_bias, enc_dec_bias

        else:
            tgt, _ = self.slf_attn(q_hid=tgt, return_position_bias=False, position_bias=dec_slf_bias)
            enc_output, enc_dec_attn = self.enc_attn(q_hid=tgt, k_hid=memory, v_hid=memory, return_position_bias=False,
                                                     position_bias=enc_dec_bias)
            enc_output = self.pos_ffn(enc_output)
            return enc_output, enc_dec_attn


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


class MultiHeadAttention(Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_kv, dropout=0.1, has_relative_attention_bias=True,
                 relative_attention_num_buckets=32):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_kv
        self.d_v = d_kv

        self.w_qs = nn.Linear(d_model, n_head * d_kv)
        self.w_ks = nn.Linear(d_model, n_head * d_kv)
        self.w_vs = nn.Linear(d_model, n_head * d_kv)

        self.attention = ScaledDotProductAttention(temperature=d_kv ** 0.5,
                                                   has_relative_attention_bias=has_relative_attention_bias,
                                                   relative_attention_num_buckets=relative_attention_num_buckets,
                                                   n_heads=self.n_head)
        self.layer_norm = LayerNorm(d_model, eps=1e-12)

        self.fc = nn.Linear(n_head * d_kv, d_model)

        self.dropout = nn.Dropout(dropout)

        # Mesh TensorFlow attention initialization to avoid scaling before softmax
        # Github: tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
        self.w_qs.weight.data.normal_(mean=0.0, std=(d_model * d_kv) ** -0.5)
        self.w_ks.weight.data.normal_(mean=0.0, std=d_model ** -0.5)
        self.w_vs.weight.data.normal_(mean=0.0, std=d_model ** -0.5)
        self.fc.weight.data.normal_(mean=0.0, std=(n_head * d_kv) ** -0.5)

    def forward(self, q_hid, k_hid=None, v_hid=None, return_position_bias=False, position_bias=None,
                context_position=None, memory_position=None, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        proj = self.layer_norm(q_hid)
        sz_b, len_q, _ = proj.size()

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(proj).view(sz_b, len_q, n_head, d_k)
        if k_hid is None:  # self-attention if kv is not available
            len_kv = len_q
            k = self.w_ks(proj).view(sz_b, len_kv, n_head, d_k)
        else:
            len_kv = k_hid.size(1)
            k = self.w_ks(k_hid).view(sz_b, len_kv, n_head, d_k)
        if v_hid is None:
            v = self.w_vs(proj).view(sz_b, len_kv, n_head, d_v)
        else:
            v = self.w_vs(v_hid).view(sz_b, len_kv, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        output_tuple = self.attention(q, k, v, return_position_bias=return_position_bias, position_bias=position_bias,
                                      context_position=context_position, memory_position=memory_position, mask=mask)
        if return_position_bias:
            output, attn, position_bias = output_tuple
        else:
            output, attn = output_tuple

        attn = torch.sum(attn, dim=1)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output)) + q_hid

        if return_position_bias:
            return output, attn, position_bias
        else:
            return output, attn


class PositionwiseFeedForward(Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.wi = nn.Linear(d_in, d_hid, bias=False)
        self.wo = nn.Linear(d_hid, d_in, bias=False)
        self.layer_norm = LayerNorm(d_in, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Mesh TensorFlow FF initialization
        # See github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
        # and github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
        self.wi.weight.data.normal_(mean=0.0, std=d_in ** -0.5)
        if hasattr(self.wi, "bias") and self.wi.bias is not None:
            self.wi.bias.data.zero_()
        self.wo.weight.data.normal_(mean=0.0, std=d_hid ** -0.5)
        if hasattr(self.wo, "bias") and self.wo.bias is not None:
            self.wo.bias.data.zero_()

    def forward(self, x):
        norm_x = self.layer_norm(x)
        output = self.wo(self.dropout1(F.gelu(self.wi(norm_x))))  # gelu - arxiv: 1606.08415
        output = x + self.dropout2(output)
        return output


class ScaledDotProductAttention(Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1, has_relative_attention_bias=True,
                 relative_attention_num_buckets=None, n_heads=None):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.has_relative_attention_bias = has_relative_attention_bias
        if self.has_relative_attention_bias:
            self.relative_attention_num_buckets = relative_attention_num_buckets
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, n_heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
        Translate relative position to a bucket number for relative attention.
        The relative position is defined as memory_position - query_position, i.e.
        the distance in tokens from the attending position to the attended-to
        position.  If bidirectional=False, then positive relative positions are
        invalid.
        We use smaller buckets for small absolute relative_position and larger buckets
        for larger absolute relative_positions.  All relative positions >=max_distance
        map to the same bucket.  All relative positions <=-max_distance map to the
        same bucket.  This should allow for more graceful generalization to longer
        sequences than the model has been trained on.
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32
            values in the range [0, num_buckets)
        """
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(torch.long) * num_buckets  # mtf.to_int32(mtf.less(n, 0)) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        # now n is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = max_exact + (
                torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def compute_bias(self, context_position, memory_position):
        """ Compute binned relative position bias """
        relative_position = memory_position[:, None, :] - context_position[:, :, None]  # shape (batch_size, qlen, klen)
        rp_bucket = self._relative_position_bucket(
            relative_position,  # shape (batch_size, qlen, klen)
            bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
        )
        rp_bucket = rp_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(rp_bucket)  # shape (batch_size, qlen, klen, num_heads)
        values = values.permute([0, 3, 1, 2])  # shape (batch_size, num_heads, qlen, klen)
        return values

    def forward(self, q, k, v, return_position_bias=False, position_bias=None, context_position=None,
                memory_position=None, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if self.has_relative_attention_bias:
            if position_bias is None:
                if return_position_bias is False:
                    raise ValueError("No position_bias returned")
                if context_position is None:
                    raise ValueError("No position_bias provided and no weights to compute position_bias")
                position_bias = self.compute_bias(context_position, memory_position)

            attn += position_bias  # (batch_size, n_heads, qlen, klen)

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        if return_position_bias:
            return output, attn, position_bias
        else:
            return output, attn


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """ Construct a layernorm module in the T5 style
            No bias and no substraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

        # init
        self.weight.data.fill_(1.0)

    def forward(self, x):
        # layer norm should always be calculated in float32
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x / torch.sqrt(variance + self.variance_epsilon)

        if self.weight.dtype == torch.float16:
            x = x.to(torch.float16)
        return self.weight * x
