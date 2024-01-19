from typing import Tuple
from math import pi, ceil, floor, log, sqrt
import logging

import torch
import torch.nn
from torch.nn.functional import one_hot


_log = logging.getLogger(__name__)


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, max_len: int):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        self.d = d_model

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, seqs: torch.Tensor, masks: torch.Tensor):

        # each seq can have a different mask, that's why we have this iteration
        for i in range(seqs.shape[0]):
            length = masks[i].sum().item()
            seq = seqs[i]
            p = torch.zeros(length, seq.shape[-1], device=seq.device)
            p[:, :] = self.pe[:length, :]
            seq[masks[i]] += p
            seqs[i] = seq

        return seqs


class RelativePositionEncoder(torch.nn.Module):

    def __init__(self, no_heads: int, relpos_k: int, depth: int):

        super(RelativePositionEncoder, self).__init__()

        self.no_heads = no_heads
        self.relpos_k = relpos_k
        self.no_bins = 2 * relpos_k + 1
        self.depth = depth
        self.inf = 1e5
        self.w_L = sqrt(1.0 / 2)  # because wee have two terms

        self.linear_q = torch.nn.Linear(depth, depth * no_heads, bias=False)
        self.linear_k = torch.nn.Linear(depth, depth * no_heads, bias=False)
        self.linear_v = torch.nn.Linear(depth, depth * no_heads, bias=False)

        self.linear_b = torch.nn.Linear(self.no_bins, self.no_heads, bias=False)

        self.linear_output = torch.nn.Linear((self.no_bins + depth) * self.no_heads, depth)

    def relpos(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            masks: [batch_size, maxlen]
                true for present residues, false for absent
        Returns:
            [*, maxlen, maxlen, no_bins] relative position codes
        """

        batch_size, maxlen = masks.shape

        # [1, 1, no_bins]
        bins = torch.arange(
            start=-self.relpos_k, end=self.relpos_k + 1,
            device=masks.device
        ).unsqueeze(0).unsqueeze(0)

        ps = []
        for batch_index, mask in enumerate(masks):

            l = mask.int().sum()

            # [l]
            r = torch.arange(l, device=mask.device)

            # [l, l, 1]
            d = (r[:, None] - r[None, :]).unsqueeze(-1)

            # [l, l, no_bins]
            p_ins = torch.nn.functional.one_hot(
                torch.argmin(torch.abs(d - bins), dim=-1),
                num_classes=self.no_bins
            ).float()

            # [max_len, max_len, no_bins]
            square_mask = (mask[:, None] * mask[None, :]).unsqueeze(-1).expand(-1, -1, self.no_bins)

            # [max_len, max_len, no_bins]
            p = torch.zeros(maxlen, maxlen, self.no_bins, device=p_ins.device)
            p[square_mask] = p_ins.view(-1)

            ps.append(p)

        return torch.stack(ps)

    def forward(self, s: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:
                features of shape [*, N_res, depth]
            masks:
                boolean of shape [*, N_res]
        Returns:
            embd: [*, N_res, depth]
            attention:  [*, H, N_res, N_res]
        """

        batch_size, maxlen = masks.shape

        # [*, N_res, N_res, no_bins]
        relpos = self.relpos(masks)

        # [*, H, N_res, N_res]
        b = self.linear_b(relpos).transpose(-2, -1).transpose(-3, -2)

        # [*, H, N_res, depth]
        q = self.linear_q(s).reshape(batch_size, maxlen, self.depth, self.no_heads).transpose(-2, -1).transpose(-3, -2)
        k = self.linear_q(s).reshape(batch_size, maxlen, self.depth, self.no_heads).transpose(-2, -1).transpose(-3, -2)
        v = self.linear_q(s).reshape(batch_size, maxlen, self.depth, self.no_heads).transpose(-2, -1).transpose(-3, -2)

        # [*, 1, N_res, N_res]
        square_mask = (masks[:, :, None] * masks[:, None, :]).unsqueeze(-3)

        # [*, H, N_res, N_res]
        a = square_mask * torch.nn.functional.softmax(
            self.w_L * (torch.matmul(q, k.transpose(-2, -1)) / sqrt(self.depth) + b),
            -1
        )

        # [*, H, N_res, no_bins]
        o_pair = (a.unsqueeze(-1) * relpos.unsqueeze(-4)).sum(-2)

        # [*, H, N_res, depth]
        o = (a.unsqueeze(-1) * v.unsqueeze(-3)).sum(-2)

        # [*, N_res, depth]
        embd = self.linear_output(
            torch.cat(
                (o_pair.transpose(-3, -2).reshape(batch_size, maxlen, self.no_heads * self.no_bins),
                 o.transpose(-3, -2).reshape(batch_size, maxlen, self.no_heads * self.depth),
                 ), dim=-1
            )
        )

        return embd, a
