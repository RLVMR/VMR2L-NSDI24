"""
This version of the transformer is for encoder only. It uses cached attention to speed up inference.
References:
1. jadore801120/attention-is-all-you-need-pytorch
2. pytorch/pytorch
"""
import copy
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.container import ModuleList


class CausalConv1d(Module):  # arxiv: 1907.00235, https://discuss.pytorch.org/t/causal-convolution/3456/3
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super(CausalConv1d, self).__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation, **kwargs)
        self.weight = self.conv1d.weight
        self.bias = self.conv1d.bias

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = x[:, :, :-self.conv1d.padding[0]]
        return x.permute(0, 2, 1)


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class TransformerEncoder(Module):

    def __init__(self, encoder_layer, num_layers, conv_size=1, d_k=20, d_v=20, num_head=8):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.conv_size = conv_size
        self.d_k = d_k
        self.d_v = d_v
        self.num_head = num_head
        self.apply(init_weights)

    def forward(self, output, mask=None, q_cache=None, k_cache=None, v_cache=None, return_attns=False,
                return_cache=False):

        if return_attns is False and return_cache is False:
            for i in range(self.num_layers):
                output, _ = self.layers[i](output, src_mask=mask)
            return output
        else:
            if k_cache is not None:
                # note that we always return one more for dim = 2 to allow for conv_size = 1
                # query is different since we only care about the output of the last timestep
                new_q_cache = torch.zeros((self.num_layers, output.shape[0], self.conv_size, output.shape[2]),
                                          device=output.device)
                new_k_cache = torch.zeros((self.num_layers, self.num_head * output.shape[0], 1, self.d_k),
                                          device=output.device)
                new_v_cache = torch.zeros((self.num_layers, self.num_head * output.shape[0], 1, self.d_v),
                                          device=output.device)
                for i in range(self.num_layers):
                    output, new_q_cache[i], new_k_cache[i], new_v_cache[i] = self.layers[i](output, src_mask=mask,
                                                                                            q_cache=q_cache[i],
                                                                                            k_cache=k_cache[i],
                                                                                            v_cache=v_cache[i],
                                                                                            return_cache=True)
                k_cache = torch.cat([k_cache, new_k_cache], dim=2)
                v_cache = torch.cat([v_cache, new_v_cache], dim=2)
                return output, new_q_cache, k_cache, v_cache
            else:
                if return_cache:
                    q_cache = torch.zeros((self.num_layers, output.shape[0], self.conv_size, output.shape[2]),
                                          device=output.device)
                    k_cache = torch.zeros((self.num_layers, self.num_head * output.shape[0], output.shape[1], self.d_k),
                                          device=output.device)
                    v_cache = torch.zeros((self.num_layers, self.num_head * output.shape[0], output.shape[1], self.d_v),
                                          device=output.device)
                    for i in range(self.num_layers):
                        output, q_cache[i], k_cache[i], v_cache[i] = self.layers[i](output, src_mask=mask,
                                                                                    return_cache=True)
                    return output, q_cache, k_cache, v_cache
                else:  # return_attns is True
                    enc_slf_attn_list = []
                    for i in range(self.num_layers):
                        output, enc_slf_attn = self.layers[i](output, src_mask=mask)
                        enc_slf_attn_list += [enc_slf_attn]
                    return output, enc_slf_attn_list[-1]


class TransformerEncoderLayer(Module):

    def __init__(self, d_model, n_head, d_k, d_v, conv_size=1, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, conv_size, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, dim_feedforward, dropout=dropout)

    def forward(self, src, src_mask=None, q_cache=None, k_cache=None, v_cache=None, return_cache=False):

        if return_cache is False:
            enc_output, enc_slf_attn = self.slf_attn(src, src, src, mask=src_mask)
            enc_output = self.pos_ffn(enc_output)
            return enc_output, enc_slf_attn
        elif k_cache is not None:
            enc_output, q_cache, k_cache, v_cache = self.slf_attn.test(src, q_cache=q_cache,
                                                                       k_cache=k_cache, v_cache=v_cache)
            enc_output = self.pos_ffn(enc_output)
            return enc_output, q_cache, k_cache, v_cache
        else:
            enc_output, q_cache, k_cache, v_cache = self.slf_attn(src, src, src, mask=src_mask, return_cache=True)
            enc_output = self.pos_ffn(enc_output)
            return enc_output, q_cache, k_cache, v_cache


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


class MultiHeadAttention(Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, conv_size, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.conv_size = conv_size

        if conv_size != 1:
            self.w_qs = CausalConv1d(d_model, n_head * d_k, conv_size)
            self.w_ks = CausalConv1d(d_model, n_head * d_k, conv_size)
        else:
            self.w_qs = nn.Linear(d_model, n_head * d_k)
            self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.fc = nn.Linear(n_head * d_v, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None, return_cache=False):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        _, len_k, _ = k.size()
        _, len_v, _ = v.size()

        residual = q
        if return_cache:
            q_cache = q[:, -self.conv_size:].clone()

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        if return_cache:
            k_cache = k.clone()
            v_cache = v.clone()

        output, attn = self.attention(q, k, v, mask=mask)

        attn = attn.view(n_head, sz_b, len_q, len_q)
        attn = torch.sum(attn, dim=0)
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output) + residual

        if return_cache:
            return output, q_cache, k_cache, v_cache
        else:
            return output, attn

    # attention with cache
    def test(self, q, q_cache=None, k_cache=None, v_cache=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()

        residual = q  # shape: torch.Size([sz_b, given_steps, d_input]
        q_copy = torch.cat([q_cache, q], dim=1)[:, 1:].view(q_cache.shape[0], self.conv_size, q_cache.shape[-1])
        new_q_cache = q_copy.clone()

        q = self.w_qs(q_copy)[:, -1].view(sz_b, 1, n_head,
                                          d_k)  # self.w_qs(q_copy) shape: torch.Size([batch_size, 1, n_head*d_k])
        k = self.w_ks(q_copy)[:, -1].view(sz_b, 1, n_head, d_k)
        v = self.w_vs(q_copy)[:, -1].view(sz_b, 1, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, 1, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, 1, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, 1, d_v)  # (n*b) x lv x dv

        new_k_cache = k.clone()
        new_v_cache = v.clone()

        k = torch.cat([k_cache, k], dim=1)
        v = torch.cat([v_cache, v], dim=1)

        output, attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, 1, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, 1, -1)  # b x lq x (n*dv)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output) + residual

        return output, new_q_cache, new_k_cache, new_v_cache


class PositionwiseFeedForward(Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.wi = nn.Linear(d_in, d_hid, bias=False)
        self.wo = nn.Linear(d_hid, d_in, bias=False)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        norm_x = self.layer_norm(x)
        output = self.wo(self.dropout1(F.gelu(self.wi(norm_x))))  # gelu - arxiv: 1606.08415
        output = x + self.dropout2(output)
        return output


class ScaledDotProductAttention(Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn