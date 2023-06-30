from typing import Optional, Tuple, Sequence
import math

import logging
import torch

from openfold.utils.rigid_utils import Rigid, Rotation
from openfold.model.primitives import Linear, LayerNorm, ipa_point_weights_init_
from openfold.utils.precision_utils import is_fp16_enabled
from openfold.utils.tensor_utils import permute_final_dims, flatten_final_dims, dict_multimap


_log = logging.getLogger(__name__)


class DebuggableInvariantPointAttention(torch.nn.Module):
    """
    Implements Algorithm 22.
    """
    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        no_qk_points: int,
        no_v_points: int,
        protein_maxlen: int,
        inf: float = 1e5,
        eps: float = 1e-8,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
        """
        super(DebuggableInvariantPointAttention, self).__init__()

        self.c_s = c_s 
        self.c_z = c_z 
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.protein_maxlen = protein_maxlen
        self.inf = inf 
        self.eps = eps 

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc, bias=False)
        self.linear_kv = Linear(self.c_s, 2 * hc, bias=False)

        self.linear_b = Linear(self.c_z, self.no_heads, bias=False)

        concat_out_dim = self.no_heads * (
            self.c_z + self.c_hidden
        )
        self.linear_out = Linear(concat_out_dim, self.c_s, init="final")

        self.softmax = torch.nn.Softmax(dim=-1)
        self.softplus = torch.nn.Softplus()

    def forward(
        self,
        s: torch.Tensor,
        z: Optional[torch.Tensor],
        r: Rigid,
        mask: torch.Tensor,
        inplace_safe: bool = False,
        _offload_inference: bool = False,
        _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
            [*, N_head, N_res, N_res] attention weights
        """
        if(_offload_inference and inplace_safe):
            z = _z_reference_list
        else:
            z = [z]

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        b = self.linear_b(z[0])

        _log.debug(f"mhc self attention: b_ij has values ranging from {b.min()} - {b.max()}")
        _log.debug(f"mhc self attention: b_ij has distribution {b.mean()} +/- {b.std()}")

        if(_offload_inference):
            assert(sys.getrefcount(z[0]) == 2)
            z[0] = z[0].cpu()

        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        general_att_mask = square_mask.clone().unsqueeze(-3)
        square_mask = (self.inf * (square_mask - 1)).unsqueeze(-3)

        # [*, H, N_res, N_res]
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

        # animation
        a_sd = a.clone() * general_att_mask

        a += (math.sqrt(1.0 / 2) * permute_final_dims(b, (2, 0, 1)))

        # animation
        a_b = math.sqrt(1.0 / 2) * permute_final_dims(b, (2, 0, 1)).clone() * general_att_mask

        if(inplace_safe):
            a += square_mask
            # in-place softmax
            attn_core_inplace_cuda.forward_(
                a,
                reduce(mul, a.shape[:-1]),
                a.shape[-1],
            )
        else:
            a = a + square_mask
            a = self.softmax(a)

        _log.debug(f"mhc self attention: a ranges {a.min()} - {a.max()}")
        _log.debug(f"mhc self attention: v ranges {v.min()} - {v.max()}")

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.matmul(
            a, v.transpose(-2, -3).to(dtype=a.dtype)
        ).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        if(_offload_inference):
            z[0] = z[0].to(o.device)

        # [*, N_res, H, C_z]
        o_pair = torch.matmul(a.transpose(-2, -3), z[0].to(dtype=a.dtype))

        # [*, N_res, H * C_z]
        o_pair = flatten_final_dims(o_pair, 2)

        _log.debug(f"mhc self attention: o ranges {o.min()} - {o.max()}")
        _log.debug(f"mhc self attention: o_pair ranges {o_pair.min()} - {o_pair.max()}")

        # [*, N_res, C_s]
        s = self.linear_out(
            torch.cat(
                (o, o_pair), dim=-1
            ).to(dtype=z[0].dtype)
        )

        return s, a.clone(), a_sd, a_b
