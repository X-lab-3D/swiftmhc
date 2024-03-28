from typing import Tuple, Optional
from math import pi, ceil, floor, log, sqrt
import logging

import torch
import torch.nn
from torch.nn.functional import one_hot


_log = logging.getLogger(__name__)


class RelativePositionEncoder(torch.nn.Module):
    """
    Gives the input sequence a relative positional encoding and performs multi-headed attention.
    """

    def __init__(
            self,
            no_heads: int, relpos_k: int, depth: int,
            dropout_rate: Optional[float] = 0.1,
            transition: Optional[int] = 128
    ):
        """
        Args:
            no_heads:       number of attention heads
            relpos_k:       determines the number of distance bins: [-k, -k + 1, ..., 0, ..., k - 1, k]
            depth:          the depth of the input tensor, at shape -1
            dropout_rate:   for the dropouts before normalisation
            transition:     transition depth in feed forward block
        """

        super(RelativePositionEncoder, self).__init__()

        # constants
        self.no_heads = no_heads
        self.relpos_k = relpos_k
        self.no_bins = 2 * relpos_k + 1
        self.depth = depth
        self.inf = 1e5
        self.w_L = sqrt(1.0 / 2)  # because we have two terms

        # scaled dot multi-headed attention: queries, keys, values
        self.linear_q = torch.nn.Linear(depth, depth * no_heads, bias=False)
        self.linear_k = torch.nn.Linear(depth, depth * no_heads, bias=False)
        self.linear_v = torch.nn.Linear(depth, depth * no_heads, bias=False)

        # generates the b term in the attention weight
        self.linear_b = torch.nn.Linear(self.no_bins, self.no_heads, bias=False)

        # generates the output of the multi-header attention
        self.linear_output = torch.nn.Linear((self.no_bins + depth) * self.no_heads, depth)

        # to be used after multi-headed attention
        self.norm_attention = torch.nn.Sequential(
            torch.nn.Dropout(dropout_rate),
            torch.nn.LayerNorm(depth),
        )

        # to be used after multi-headed attention norm
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(depth, transition),
            torch.nn.ReLU(),
            torch.nn.Linear(transition, depth),
        )

        # to be used after feed-forward
        self.norm_feed_forward = torch.nn.Sequential(
            torch.nn.Dropout(dropout_rate),
            torch.nn.LayerNorm(depth),
        )

    def relpos(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Creates a relative position matrix.

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

        x, a = self.attention(s, masks)
        s = self.norm_attention(s + x)

        x = self.feed_forward(s)
        s = self.norm_feed_forward(s + x)

        return s, a

    def attention(self, s: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs multi-headed attention.

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
        square_mask = torch.logical_and(masks[:, :, None], masks[:, None, :]).unsqueeze(-3)

        # [*, H, N_res, N_res]
        a = torch.nn.functional.softmax(
            self.w_L * (torch.matmul(q, k.transpose(-2, -1)) / sqrt(self.depth) + b) - self.inf * torch.logical_not(square_mask).float(),
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
