import math
import logging

import torch

from openfold.utils.precision_utils import is_fp16_enabled
from openfold.utils.rigid_utils import Rigid
from openfold.model.primitives import Linear, LayerNorm, ipa_point_weights_init_
from openfold.utils.tensor_utils import (
    permute_final_dims,
    flatten_final_dims,
)


_log = logging.getLogger(__name__)


class CrossInvariantPointAttention(torch.nn.Module):
    def __init__( self,
        c_s: int,
        c_hidden: int,
        no_heads: int,
        no_qk_points: int,
        no_v_points: int,
        eps: float = 1e-8,
        inf: float = 1e9,
    ):
        """
        Args:
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

        self.c_s = c_s
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.eps = eps

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

        self.inf = inf

        self.head_weights = torch.nn.Parameter(torch.zeros((no_heads)))
        ipa_point_weights_init_(self.head_weights)

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
        inplace_safe: bool = False,
        _offload_inference: bool = False,

    ) -> torch.Tensor:
        """
        Args:
            s_dst:
                [batch_size, len_dst, c_s] single representation
            s_src:
                [batch_size, len_src, c_s] single representation
            T_dst:
                [batch_size, len_dst, 4, 4] transformation object
            T_src:
                [batch_size, len_src, 4, 4] transformation object
            dst_mask:
                [batch_size, len_dst] booleans
            src_mask:
                [batch_size, len_src] booleans
        Returns:
            [batch_size, len_dst, c_s] single representation update
        """

        _log.debug(f"cross: s_dst ranges {s_dst.min()} - {s_dst.max()}")
        _log.debug(f"cross: s_src ranges {s_src.min()} - {s_src.max()}")

        # [batch_size, 1, len_dst, len_src]
        general_att_mask = (dst_mask.unsqueeze(-1) * src_mask.unsqueeze(-2)).unsqueeze(-3)

        #######################################
        # Generate scalar and point activations
        #######################################
        # [batch_size, len_dst, H * C_hidden]
        q = self.linear_q(s_dst)

        # [batch_size, len_src, H * C_hidden]
        kv = self.linear_kv(s_src)

        # [batch_size, len_dst, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [batch_size, len_src, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [batch_size, len_src, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # [batch_size, len_dst, H * P_q * 3]
        q_pts = self.linear_q_points(s_dst)

        # This is kind of clunky, but it's how the original does it
        # [batch_size, len_dst, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = T_dst[..., None].apply(q_pts)

        # [batch_size, len_dst, H, P_q, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
        )

        # [batch_size, len_dst, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s_src)

        # [batch_size, len_src, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = T_src[..., None].apply(kv_pts)

        # [batch_size, len_src, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # [batch_size, len_src, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )

        ##########################
        # Compute attention scores : line #7 in alphafold
        ##########################

        if(is_fp16_enabled()):
            with torch.cuda.amp.autocast(enabled=False):
                a = torch.matmul(
                    permute_final_dims(q.float(), (1, 0, 2)),  # [*, H, N_res, C_hidden]
                    permute_final_dims(k.float(), (1, 2, 0)),  # [*, H, C_hidden, N_res]
                )
        else:
            a = torch.matmul(
                permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
                permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
            )

        a *= math.sqrt(1.0 / (2 * self.c_hidden))

        # animate
        a_sd = a * general_att_mask

        # [batch_size, len_dst, len_src, H, P_q, 3]
        pt_att = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        if(inplace_safe):
            pt_att *= pt_att
        else:
            pt_att = pt_att ** 2

        # [batch_size, len_dst, len_src, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))

        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )

        # only two terms in attention weight, so divide by two
        head_weights = head_weights * math.sqrt(
            1.0 / (2 * (self.no_qk_points * 9.0 / 2))
        )
        if(inplace_safe):
            pt_att *= head_weights
        else:
            pt_att = pt_att * head_weights

        # [batch_size, len_dst, len_src, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)

        # [batch_size, len_dst, len_src]
        square_mask = dst_mask.unsqueeze(-1) * src_mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask.to(dtype=torch.float32) - 1.0)

        # [batch_size, H, len_dst, len_src]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))

        # animate
        a_pts = pt_att * general_att_mask

        if(inplace_safe):
            a += pt_att
            del pt_att
            a += square_mask.unsqueeze(-3)
            # in-place softmax
            attn_core_inplace_cuda.forward_(
                a,
                reduce(mul, a.shape[:-1]),
                a.shape[-1],
            )
        else:
            a = a + pt_att
            a = a + square_mask.unsqueeze(-3)
            a = self.softmax(a)

        ################
        # Compute output
        ################
        # [batch_size, len_dst, H, C_hidden]
        o = torch.matmul(
            a, v.transpose(-2, -3).to(dtype=a.dtype)
        ).transpose(-2, -3)

        # [batch_size, len_dst, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [batch_size, H, 3, len_dst, P_v]
        if(inplace_safe):
            v_pts = permute_final_dims(v_pts, (1, 3, 0, 2))
            o_pt = [
                torch.matmul(a, v.to(a.dtype))
                for v in torch.unbind(v_pts, dim=-3)
            ]
            o_pt = torch.stack(o_pt, dim=-3)
        else:
            o_pt = torch.sum(
                (
                    a[..., None, :, :, None]
                    * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
                ),
                dim=-2,
            )

        # [batch_size, len_dst, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = T_dst[..., None, None].invert_apply(o_pt)

        # [batch_size, len_dst, H * P_v]
        o_pt_norm = flatten_final_dims(
            torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps), 2
        )

        # [batch_size, len_dst, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        # [batch_size, len_dst, c_s]
        s_upd = self.linear_out(
            torch.cat(
                (o, *torch.unbind(o_pt, dim=-1), o_pt_norm), dim=-1
            ).to(dtype=s_dst.dtype)
        )

        return s_upd, a, a_sd, a_pts

