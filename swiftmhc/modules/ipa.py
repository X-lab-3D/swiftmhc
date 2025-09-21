import logging
from math import sqrt
import ml_collections
import torch
from openfold.model.primitives import Linear
from openfold.utils.tensor_utils import flatten_final_dims
from openfold.utils.tensor_utils import permute_final_dims
from ..tools.rigid import Rigid


_log = logging.getLogger(__name__)


class DebuggableInvariantPointAttention(torch.nn.Module):
    def __init__(self, config: ml_collections.ConfigDict):
        """This is like Algorithm 22 in AlphaFold2, but without the third attention weight term.
        Also, the second attention weight term is made from a proximity matrix.
        The code was taken from OpenFold and then modified.
        The original InvariantPointAttention class is at:
        https://github.com/aqlaboratory/openfold/blob/main/openfold/model/structure_module.py

        in config:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            num_heads:
                Number of attention heads
            num_qk_points:
                Number of query/key points to generate
            num_v_points:
                Number of value points to generate
        """
        super(DebuggableInvariantPointAttention, self).__init__()

        self.c_s = config.c_s
        self.c_z = config.c_z
        self.c_hidden = config.c_hidden
        self.num_heads = config.num_heads
        self.num_qk_points = config.num_qk_points
        self.num_v_points = config.num_v_points
        self.inf = config.inf
        self.eps = config.epsilon

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.num_heads
        self.linear_q = Linear(self.c_s, hc, bias=False)
        self.linear_kv = Linear(self.c_s, 2 * hc, bias=False)

        self.linear_b = Linear(self.c_z, self.num_heads, bias=False)

        concat_out_dim = self.num_heads * (self.c_z + self.c_hidden)
        self.linear_out = Linear(concat_out_dim, self.c_s, init="final")

        self.softmax = torch.nn.Softmax(dim=-1)
        self.softplus = torch.nn.Softplus()

        # we have two attention terms, so divide by two
        self.w_L = sqrt(0.5)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor | None,
        r: Rigid,
        mask: torch.Tensor,
        inplace_safe: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs Invariant Point attention on the residues within one sequence.

        Args:
            s:      [*, N_res, C_s] single representation
            z:      [*, N_res, N_res, C_z] pair representation
            r:      [*, N_res] transformation object
            mask:   [*, N_res] mask
        Returns:
            [*, N_res, C_s] updated single representation
            [*, H, N_res, N_res] attention weights
        """
        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.num_heads, -1))

        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.num_heads, -1))

        # [*, N_res, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        b = self.linear_b(z)

        # [*, H, N_res, N_res]
        b = permute_final_dims(b, (2, 0, 1))

        # [*, N_res, N_res]
        square_mask = torch.logical_and(mask.unsqueeze(-1), mask.unsqueeze(-2))

        # [*, H, N_res, N_res]
        a_sd = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        ) / sqrt(self.c_hidden)

        a = self.softmax(
            self.w_L * (a_sd + b)
            - self.inf * torch.logical_not(square_mask[..., None, :, :]).to(dtype=s.dtype)
        )

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.matmul(a, v.transpose(-2, -3).to(dtype=a.dtype)).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, N_res, H, C_z]
        o_pair = torch.matmul(a.transpose(-2, -3), z.to(dtype=a.dtype))

        # [*, N_res, H * C_z]
        o_pair = flatten_final_dims(o_pair, 2)

        # [*, N_res, C_s]
        s_upd = self.linear_out(torch.cat((o, o_pair), dim=-1).to(dtype=z.dtype))

        return s_upd, a
