"""
Implementation of the Gated Residual Network (GRN) and Variable Selection Network (VSN).
References:
1. https://arxiv.org/pdf/1912.09363.pdf
"""

import warnings
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.module import Module
from typing import List, Optional, Tuple
from .transformer import LayerNorm


class GRN(Module):

    def __init__(self, d_input, d_hidden, dropout, d_output=None):
        super(GRN, self).__init__()
        if d_output is None:
            d_output = d_hidden
            self.skip = nn.Identity()
        else:
            self.skip = nn.Linear(d_input, d_output, bias=True)
        self.linear_elu = nn.Linear(d_input, d_hidden, bias=True)
        self.linear_pre_glu = nn.Linear(d_hidden, d_hidden, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_post_glu = nn.Linear(d_hidden, 2 * d_output, bias=True)
        self.layer_norm = LayerNorm(d_output, eps=1e-6)

    def forward(self, alpha, context=None):

        if context is not None:
            together = torch.cat((alpha, context), dim=-1)
        else:
            together = alpha
        post_elu = F.elu(self.linear_elu(together))
        pre_glu = self.dropout(self.linear_pre_glu(post_elu))
        return self.layer_norm(F.glu(self.linear_post_glu(pre_glu)) + self.skip(alpha))


# optional dropout and Gated Liner Unit followed by add and norm
class AddNorm(Module):
    def __init__(self, d_hidden, dropout):
        super(AddNorm, self).__init__()
        self.linear_glu = nn.Linear(d_hidden, 2 * d_hidden, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(d_hidden, eps=1e-6)

    def forward(self, x, residual):
        post_glu = F.glu(self.linear_glu(self.dropout(x)))
        return self.layer_norm(post_glu + residual)


# referencing Xiaoyong's code
class VSN(Module):
    def __init__(self, d_hidden: int, n_vars: int, dropout: float = 0.0, add_static: bool = False):
        super(VSN, self).__init__()
        self.add_static = add_static
        self.d_hidden = d_hidden
        self.n_vars = n_vars
        self.weight_network = GRN(
            d_input=self.d_hidden * (self.n_vars+int(self.add_static)),
            d_hidden=self.d_hidden,
            dropout=dropout,
            d_output=self.n_vars,
        )
        self.variable_network = nn.ModuleList(
            GRN(
                d_input=self.d_hidden,
                d_hidden=self.d_hidden,
                dropout=dropout,
            ) for _ in range(n_vars)
        )

    def forward(self, variables: List[Tensor], static_encoding: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        if len(variables) != self.n_vars:
            raise ValueError(f'Expected {self.n_vars} variables, but {len(variables)} given.')
        if self.add_static and static_encoding is None:
            raise ValueError('Static variable encoding is required.')
        if not self.add_static and static_encoding is not None:
            warnings.warn('Static variable encoding is not expected but given. Ignored.')

        # [B, *, d_hidden*n_vars] / [B, *, d_hidden*(n_vars+1)]
        flatten = torch.cat(variables, dim=-1)
        if self.add_static and static_encoding is not None:
            static_encoding = static_encoding.unsqueeze(dim=1).repeat(1, flatten.shape[1], 1)
            flatten = torch.cat([flatten, static_encoding], dim=-1)
        # [B, *, 1, n_vars]
        weight = self.weight_network(flatten).unsqueeze(dim=-2)
        weight = torch.softmax(weight, dim=-1)
        # [B, *, d_hidden, n_vars]
        var_encodings = torch.stack(
            tensors=[net(v) for v, net in zip(variables, self.variable_network)],
            dim=-1
        )
        var_encodings = torch.sum(var_encodings * weight, dim=-1)
        return var_encodings, weight
