from typing import Tuple
from math import sqrt
import logging

import torch

import ml_collections

from openfold.utils.precision_utils import is_fp16_enabled
from openfold.model.primitives import Linear, LayerNorm, ipa_point_weights_init_
from openfold.utils.tensor_utils import (
    permute_final_dims,
    flatten_final_dims,
)
from ..tools.rigid import Rigid


_log = logging.getLogger(__name__)


class CrossInvariantPointAttention(torch.nn.Module):
    def __init__(self, config: ml_collections.ConfigDict):

        """
        This is like Algorithm 22 in AlphaFold2, but between two different sequences: a source and destination.
        The idea is that the source is used to update the destination sequence.
        The second term was removed from the attention weight formula.
        The code was taken from OpenFold and then modified.

        in config:
            c_s:
                Single representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
        """
        super(CrossInvariantPointAttention, self).__init__()

        self.c_s = config.c_s
        self.c_hidden = config.c_hidden
        self.no_heads = config.no_heads
        self.no_qk_points = config.no_qk_points
        self.no_v_points = config.no_v_points
        self.eps = config.epsilon
        self.inf = config.inf

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc, bias=False)
        self.linear_kv = Linear(self.c_s, 2 * hc, bias=False)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq, bias=False)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv, bias=False)

        hpv = self.no_heads * self.no_v_points * 3

        concat_out_dim = self.no_heads * (
            self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = Linear(concat_out_dim, self.c_s, init="final")

        self.softmax = torch.nn.Softmax(dim=-1)
        self.softplus = torch.nn.Softplus()

        # one weight per head: [H]
        self.head_weights = torch.nn.Parameter(torch.zeros((self.no_heads)))
        ipa_point_weights_init_(self.head_weights)

        self.w_C = sqrt(2.0 / (9.0 * self.no_qk_points))

        # only two terms in attention weight, so divide by two
        self.w_L = sqrt(0.5)

    @staticmethod
    def _standardize_pts_attention(a: torch.Tensor) -> torch.Tensor:

        dims = (1, 2, 3)

        m = a.mean(dim=dims)
        s = a.std(dim=dims)

        return (a - m[:, None, None, None]) / s[:, None, None, None]

    def forward(
        self,
        s_dst: torch.Tensor,
        s_src: torch.Tensor,
        T_dst: Rigid,
        T_src: Rigid,
        dst_mask: torch.Tensor,
        src_mask: torch.Tensor,

    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform Invariant Point Attention between two sequences.
        Update the destination(dst) sequence from the source(src) sequence.

        Args:
            s_dst:
                [*, len_dst, c_s] single representation
            s_src:
                [*, len_src, c_s] single representation
            T_dst:
                [*, len_dst] transformation object
            T_src:
                [*, len_src] transformation object
            dst_mask:
                [*, len_dst] booleans
            src_mask:
                [*, len_src] booleans
        Returns:
            [*, len_dst, c_s] updated single representation
            [*, H, len_dst, len_src] attention weights
        """

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, len_dst, H * C_hidden]
        q = self.linear_q(s_dst)

        # [*, len_src, H * C_hidden]
        kv = self.linear_kv(s_src)

        # [*, len_dst, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, len_src, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, len_src, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # [*, len_dst, H * P_q * 3]
        q_pts = self.linear_q_points(s_dst)

        # This is kind of clunky, but it's how the original does it
        # [*, len_dst, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = T_dst[..., None].apply(q_pts)

        # [*, len_dst, H, P_q, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
        )

        # [*, len_dst, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s_src)

        # [*, len_src, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = T_src[..., None].apply(kv_pts)

        # [*, len_src, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # [*, len_src, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )

        ##########################
        # Compute attention scores : line #7 in alphafold
        ##########################

        # [*, H, len_dst, len_src]
        a_sd = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, len_dst, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, len_src]

        ) * sqrt(1.0 / self.c_hidden)

        # only two terms in attention weight, so divide by two
        # [1, 1, 1, H]
        head_weights = self.softplus(self.head_weights).view([1] * len(q_pts.shape[:-3]) + [1, -1])

        # [*, len_dst, len_src, H]
        a_pt = (0.5 * head_weights * self.w_C) * ((q_pts[..., :, None, :, :, :] - k_pts[..., None, :, :, :, :]) ** 2).sum(dim=(-2, -1))

        # [*, H, len_dst, len_src]
        a_pt = permute_final_dims(a_pt, (2, 0, 1))

        # [*, len_dst, len_src]
        square_mask = torch.logical_and(dst_mask.unsqueeze(-1), src_mask.unsqueeze(-2))

        # [*, H, len_dst, len_src]
        a = self.softmax(self.w_L * (a_sd - a_pt) - self.inf * torch.logical_not(square_mask).to(dtype=s_dst.dtype)[..., None, :, :])

        ################
        # Compute output
        ################
        # [*, len_dst, H, C_hidden]
        o = torch.matmul(
            a, v.transpose(-2, -3).to(dtype=a.dtype)
        ).transpose(-2, -3)

        # [*, len_dst, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, H, 3, len_dst, P_v]
        o_pt = torch.sum(
            a[..., None, :, :, None] * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :],
            dim=-2,
        )

        # [*, len_dst, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = T_dst[..., None, None].invert_apply(o_pt)

        # [*, len_dst, H * P_v]
        o_pt_norm = flatten_final_dims(
            torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps), 2
        )

        # [*, len_dst, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        # [*, len_dst, c_s]
        s_upd = self.linear_out(
            torch.cat(
                (o, *torch.unbind(o_pt, dim=-1), o_pt_norm), dim=-1
            ).to(dtype=s_dst.dtype)
        )

        return s_upd, a

