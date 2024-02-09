from typing import Optional
import logging
from math import sqrt

import torch

from torch.nn.modules.transformer import TransformerEncoderLayer


_log = logging.getLogger(__name__)



class DebuggableTransformerEncoderLayer(torch.nn.Module):
    def __init__(self,
                 depth: int,
                 n_head: int,
                 dropout: Optional[float] = 0.1):

        super(DebuggableTransformerEncoderLayer, self).__init__()

        self.n_head = n_head
        self.inf = 1e9

        self.dropout = torch.nn.Dropout(dropout)

        self.linear_q = torch.nn.Linear(depth, depth * self.n_head, bias=False)
        self.linear_k = torch.nn.Linear(depth, depth * self.n_head, bias=False)
        self.linear_v = torch.nn.Linear(depth, depth * self.n_head, bias=False)

        self.linear_o = torch.nn.Linear(self.n_head, 1, bias=False)

        self.norm_att = torch.nn.LayerNorm(depth)

        self.ff_intermediary_depth = 128

        self.mlp_ff = torch.nn.Sequential(
            torch.nn.Linear(depth, self.ff_intermediary_depth),
            torch.nn.ReLU(),
            torch.nn.Linear(self.ff_intermediary_depth, depth),
        )

        self.norm_ff = torch.nn.LayerNorm(depth)

    def self_attention(
        self,
        seq: torch.Tensor,
        seq_mask: torch.Tensor
    ) -> torch.Tensor:

        batch_size, seq_len, d = seq.shape

        # [batch_size, n_head, seq_len, d]
        q = self.linear_q(seq).reshape(batch_size, seq_len, self.n_head, d).transpose(1, 2)
        k = self.linear_k(seq).reshape(batch_size, seq_len, self.n_head, d).transpose(1, 2)
        v = self.linear_v(seq).reshape(batch_size, seq_len, self.n_head, d).transpose(1, 2)

        # [batch_size, 1, seq_len, seq_len]
        mask = torch.logical_and(seq_mask[:, None, :, None], seq_mask[:, None, None, :])
        mask = torch.logical_not(mask).float() * -self.inf

        # [batch_size, n_head, seq_len, seq_len]
        a = torch.softmax(torch.matmul(q, k.transpose(2, 3)) / sqrt(d) + mask, dim=3)

        # [batch_size, n_head, seq_len, d]
        heads = torch.matmul(a, v)

        # [batch_size, seq_len, d]
        o = self.linear_o(heads.transpose(1, 2).transpose(2, 3))[...,0]

        return o, a

    def feed_forward(self, seq: torch.Tensor) -> torch.Tensor:

        o = self.mlp_ff(seq)

        return o

    def forward(self,
                seq: torch.Tensor,
                seq_mask: torch.Tensor) -> torch.Tensor:

        x = seq

        x = self.dropout(x)

        y, a = self.self_attention(x, seq_mask)

        y = self.dropout(y)
        x = self.norm_att(x + y)

        y = self.feed_forward(x)

        y = self.dropout(y)
        x = self.norm_ff(x + y)

        return x, a


