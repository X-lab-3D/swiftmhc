import logging
from math import sqrt
from typing import Tuple
import ml_collections
import torch
import torch.nn
from openfold.model.primitives import LayerNorm
from openfold.model.primitives import Linear
from position_encoding.relative import get_relative_position_encoding_matrix


_log = logging.getLogger(__name__)


class PeptideSelfAttention(torch.nn.Module):
    """Gives the input sequence a relative positional encoding and performs multi-headed attention."""

    def __init__(self, config: ml_collections.ConfigDict):
        """In config:
        no_heads:           number of attention heads
        peptide_maxlen(k):  determines the number of distance bins: [-k, -k + 1, ..., 0, ..., k - 1, k]
        c_s:                the depth of the input tensor, at shape -1
        dropout_rate:       for the dropouts before normalisation
        c_transition:       transition depth in feed forward block
        c_hidden:           the depth of the hidden tensors: attention query(q), keys(k), values(v)
        """
        super(PeptideSelfAttention, self).__init__()

        # constants
        self.no_heads = config.no_heads
        self.relpos_k = config.peptide_maxlen
        self.no_bins = 2 * self.relpos_k + 1
        self.c_s = config.c_s
        self.c_hidden = config.c_hidden
        self.inf = config.inf
        self.w_L = sqrt(1.0 / 2)  # because we have two terms
        self.dropout_rate = config.dropout_rate
        self.c_transition = config.c_transition

        # scaled dot multi-headed attention: queries, keys, values
        self.linear_q = Linear(self.c_s, self.c_hidden * self.no_heads, bias=False)
        self.linear_k = Linear(self.c_s, self.c_hidden * self.no_heads, bias=False)
        self.linear_v = Linear(self.c_s, self.c_hidden * self.no_heads, bias=False)

        # generates the b term in the attention weight
        self.linear_b = Linear(self.no_bins, self.no_heads, bias=False)

        # generates the output of the multi-headed attention
        self.linear_output = Linear(
            (self.no_bins + self.c_hidden) * self.no_heads, self.c_s, init="final"
        )

        # to be used after multi-headed attention
        self.norm1 = torch.nn.Sequential(
            torch.nn.Dropout(self.dropout_rate),
            LayerNorm(self.c_s),
        )

        # to be used after multi-headed attention norm
        self.feed_forward = torch.nn.Sequential(
            Linear(self.c_s, self.c_transition, init="relu"),
            torch.nn.ReLU(),
            Linear(self.c_transition, self.c_s, init="final"),
        )

        # to be used after feed-forward
        self.norm2 = torch.nn.Sequential(
            torch.nn.Dropout(self.dropout_rate),
            LayerNorm(self.c_s),
        )

    def forward(self, s: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes a sequence, by means of self attention and feed forward MLP

        Args:
            s:      [*, N_res, c_s]
            mask:   [*, N_res] (bool)

        Returns:
            updated s:  [*, N_res, c_s]
            attention:  [*, H, N_res, N_res]
        """
        s_upd, a = self.attention(s, mask)
        s = self.norm1(s + s_upd)

        s = self.norm2(s + self.feed_forward(s))

        return s, a

    def attention(self, s: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs multi-headed attention, but also takes relative positions into account.

        Args:
            s:      [*, N_res, c_s]
            mask:   [*, N_res] (bool)

        Returns:
            updated s:  [*, N_res, c_s]
            attention:  [*, H, N_res, N_res]
        """
        batch_size, maxlen = mask.shape

        # [*, N_res, N_res]
        square_mask = torch.logical_and(mask[..., :, None], mask[..., None, :])

        # [*, N_res, N_res, no_bins]
        z = get_relative_position_encoding_matrix(maxlen, self.no_bins).to(
            device=s.device, dtype=s.dtype
        )
        z = z[None, ...] * square_mask[..., None]

        # [*, H, N_res, N_res]
        b = self.linear_b(z).transpose(-2, -1).transpose(-3, -2)

        # [*, H, N_res, c_hidden]
        q = (
            self.linear_q(s)
            .reshape(batch_size, maxlen, self.c_hidden, self.no_heads)
            .transpose(-2, -1)
            .transpose(-3, -2)
        )
        k = (
            self.linear_k(s)
            .reshape(batch_size, maxlen, self.c_hidden, self.no_heads)
            .transpose(-2, -1)
            .transpose(-3, -2)
        )
        v = (
            self.linear_v(s)
            .reshape(batch_size, maxlen, self.c_hidden, self.no_heads)
            .transpose(-2, -1)
            .transpose(-3, -2)
        )

        # [*, H, N_res, N_res]
        a = torch.nn.functional.softmax(
            self.w_L * (torch.matmul(q, k.transpose(-2, -1)) / sqrt(self.c_hidden) + b)
            - self.inf * torch.logical_not(square_mask[..., None, :, :]).to(dtype=s.dtype),
            dim=-1,
        )

        # [*, H, N_res, no_bins]
        o_pair = (a.unsqueeze(-1) * z.unsqueeze(-4)).sum(-2)

        # [*, H, N_res, c_hidden]
        o = (a.unsqueeze(-1) * v.unsqueeze(-3)).sum(-2)

        # [*, N_res, c_s]
        embd = self.linear_output(
            torch.cat(
                (
                    o_pair.transpose(-3, -2).reshape(
                        batch_size, maxlen, self.no_heads * self.no_bins
                    ),
                    o.transpose(-3, -2).reshape(batch_size, maxlen, self.no_heads * self.c_hidden),
                ),
                dim=-1,
            )
        )

        return embd, a
