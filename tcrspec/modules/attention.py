from typing import Optional, Tuple
import logging
from math import sqrt

import torch

from torch.nn.modules.transformer import TransformerEncoderLayer


_log = logging.getLogger(__name__)



class DebuggableCrossAttention(torch.nn.Module):
    def __init__(self,
                 depth: int,
                 n_head: int):

        super(DebuggableCrossAttention, self).__init__()

        self.n_head = n_head
        self.inf = 1e9

        self.linear_q = torch.nn.Linear(depth, depth * self.n_head, bias=False)
        self.linear_k = torch.nn.Linear(depth, depth * self.n_head, bias=False)
        self.linear_v = torch.nn.Linear(depth, depth * self.n_head, bias=False)

        self.linear_o = torch.nn.Linear(self.n_head * depth, depth, bias=False)

    def forward(
        self,
        seq1: torch.Tensor,
        seq1_mask: torch.Tensor,
        seq2: torch.Tensor,
        seq2_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, seq1_len, d = seq1.shape
        seq2_len = seq2.shape[1]

        # [batch_size, n_head, seq_len1, d]
        q = self.linear_q(seq1).reshape(batch_size, seq1_len, self.n_head, d).transpose(1, 2)
        # [batch_size, n_head, seq_len2, d]
        k = self.linear_k(seq2).reshape(batch_size, seq2_len, self.n_head, d).transpose(1, 2)
        v = self.linear_v(seq2).reshape(batch_size, seq2_len, self.n_head, d).transpose(1, 2)

        # [batch_size, 1, seq1_len, seq2_len]
        mask = torch.logical_and(seq1_mask[:, None, :, None], seq2_mask[:, None, None, :])
        mask = torch.logical_not(mask).float() * -self.inf

        # [batch_size, n_head, seq1_len, seq2_len]
        a = torch.softmax(torch.matmul(q, k.transpose(2, 3)) / sqrt(d) + mask, dim=3)

        # [batch_size, n_head, seq_len, d]
        heads = torch.matmul(a, v)

        # [batch_size, seq_len, d]
        o = self.linear_o(heads.transpose(1, 2).reshape(batch_size, seq1_len, d * self.n_head))

        return o, a

