from typing import Dict, Union, Tuple
from copy import deepcopy as copy
import logging
import sys
from math import sqrt

from torch.nn import Embedding
from torch.nn.modules.transformer import TransformerEncoder
import torch
import ml_collections

from openfold.utils.rigid_utils import Rigid, Rotation
from openfold.utils.loss import compute_plddt
from openfold.data.data_transforms import make_atom14_masks
from openfold.data.data_pipeline import make_sequence_features
from openfold.utils.loss import find_structural_violations
from openfold.utils.feats import atom14_to_atom37
from openfold.model.primitives import LayerNorm

from ..models.types import ModelType
from .position_encoding import get_relative_position_encoding
from .cross_structure_module import CrossStructureModule
from ..domain.amino_acid import AMINO_ACID_DIMENSION
from ..models.data import TensorDict
from ..tools.amino_acid import one_hot_decode_sequence
from .transform import DebuggableTransformerEncoderLayer
from .ipa import DebuggableInvariantPointAttention as IPA


_log = logging.getLogger(__name__)


class Predictor(torch.nn.Module):
    def __init__(self,
                 model_type: ModelType,
                 loop_maxlen: int,
                 protein_maxlen: int,
                 config: ml_collections.ConfigDict):
        super(Predictor, self).__init__()

        structure_module_config = copy(config.structure_module)
        structure_module_config.c_s = 32
        structure_module_config.c_z = 1
        structure_module_config.no_blocks = 2
        structure_module_config.no_heads_ipa = 2

        self.model_type = model_type
        self.loop_maxlen = loop_maxlen
        self.protein_maxlen = protein_maxlen

        self.n_head = structure_module_config.no_heads_ipa

        self.position_encoding_depth = 32

        transition_depth = 128

        loop_multihead_dim = structure_module_config.c_s * self.n_head

        self.linear_b = torch.nn.Linear(self.position_encoding_depth, self.n_head, bias=False)

        self.linear_q = torch.nn.Linear(structure_module_config.c_s, loop_multihead_dim, bias=False)
        self.linear_k = torch.nn.Linear(structure_module_config.c_s, loop_multihead_dim, bias=False)
        self.linear_v = torch.nn.Linear(structure_module_config.c_s, loop_multihead_dim, bias=False)
        self.linear_o = torch.nn.Linear(loop_multihead_dim, structure_module_config.c_s, bias=False)

        self.loop_dropout = torch.nn.Dropout(p=0.1)
        self.loop_norm = torch.nn.LayerNorm((self.loop_maxlen, structure_module_config.c_s))
        self.loop_transition = torch.nn.Sequential(
            torch.nn.Linear(structure_module_config.c_s, transition_depth),
            torch.nn.ReLU(),
            torch.nn.Linear(transition_depth, transition_depth),
            torch.nn.ReLU(),
            torch.nn.Linear(transition_depth, structure_module_config.c_s),
            torch.nn.LayerNorm(structure_module_config.c_s),
            torch.nn.Dropout(p=0.1)
        )

        self.n_block = structure_module_config.no_blocks

        self.protein_prox_norm = torch.nn.LayerNorm((self.protein_maxlen, self.protein_maxlen, 1))

        self.inf = 1e22

        self.protein_ipa = IPA(structure_module_config.c_s,
                               structure_module_config.c_z,
                               structure_module_config.c_ipa,
                               structure_module_config.no_heads_ipa,
                               structure_module_config.no_qk_points,
                               structure_module_config.no_v_points,
                               self.protein_maxlen)
        self.protein_ipa.inf = self.inf

        self.protein_norm = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            torch.nn.LayerNorm((self.protein_maxlen, structure_module_config.c_s))
        )

        self.cross = CrossStructureModule(**structure_module_config)

        self.aff_dropout = torch.nn.Dropout(p=0.1)
        self.aff_norm = torch.nn.LayerNorm((self.loop_maxlen, structure_module_config.c_s))

        self.aff_trans = torch.nn.Sequential(
            torch.nn.Linear(structure_module_config.c_s, transition_depth),
            torch.nn.ReLU(),
            torch.nn.Linear(transition_depth, transition_depth),
            torch.nn.ReLU(),
            torch.nn.Linear(transition_depth, 1),
        )

        if self.model_type == ModelType.REGRESSION:
            output_size = 1

        elif self.model_type == ModelType.CLASSIFICATION:
            output_size = 2

        self.output_linear = torch.nn.Linear(self.loop_maxlen, output_size)

    def _loop_self_attention(self,
        loop_embd: torch.Tensor,
        loop_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, loop_maxlen, loop_depth = loop_embd.shape

        # positional encoding
        # [batch_size, n_head, loop_len, loop_len]
        b_loop = self.linear_b(
            get_relative_position_encoding(loop_mask, self.position_encoding_depth)
        ).reshape(batch_size, loop_maxlen, loop_maxlen, self.n_head).transpose(1, 3)

        # self attention on the loop
        # [batch_size, loop_len, loop_len]
        loop_sqr_mask = torch.logical_and(loop_mask.unsqueeze(-2), loop_mask.unsqueeze(-1))
        loop_sqr_mask = torch.logical_not(loop_sqr_mask).float() * -self.inf

        attentions = []
        for block_index in range(self.n_block):

            # [batch_size, loop_len, n_head, embed_dim]
            q_loop = self.linear_q(loop_embd).reshape(batch_size, loop_maxlen, self.n_head, -1)
            k_loop = self.linear_k(loop_embd).reshape(batch_size, loop_maxlen, self.n_head, -1)
            v_loop = self.linear_v(loop_embd).reshape(batch_size, loop_maxlen, self.n_head, -1)

            embed_dim = q_loop.shape[-1]

            loop_heads = []

            attentions.append([])
            for head_index in range(self.n_head):

                # [batch_size, loop_len, loop_len]
                a = torch.softmax(
                    torch.bmm(
                        q_loop[..., head_index, :],
                        k_loop[..., head_index, :].transpose(-2, -1)
                    ) / sqrt(embed_dim) + loop_sqr_mask + b_loop[:, head_index, ...],
                dim=-1)

                # [batch_size, loop_len, embed_dim]
                loop_heads.append(torch.bmm(a, v_loop[..., head_index, :]))

                attentions[block_index].append(a.detach())

            # [batch_size, loop_len, n_head, embed_dim]
            loop_heads = torch.stack(loop_heads).transpose(0, 1).transpose(1, 2)

            # [batch_size, loop_len, c_s]
            loop_embd = self.linear_o(loop_heads.reshape(batch_size, loop_maxlen, -1))

        # [batch_size, n_block, n_head, loop_len, loop_len]
        attentions = torch.stack([torch.stack(a) for a in attentions]).transpose(1, 2).transpose(0, 1)

        return loop_embd, attentions

    def forward(self, batch: TensorDict) -> TensorDict:
        """
            Returns:
                single:               [batch_size, loop_len, c_s]
                aatype:               [batch_size, loop_len]  (int)
                affinity:             [batch_size]

                frames:               [n_blocks, batch_size, loop_len, 4, 4]
                sidechain_frames:     [n_blocks, batch_size, loop_len, 8, 4, 4]
                unnormalized_angles:  [n_blocks, batch_size, loop_len, 7, 2]
                angles:               [n_blocks, batch_size, loop_len, 7, 2]
                positions:            [n_blocks, batch_size, loop_len, 14, 3]
                states:               [n_blocks, batch_size, loop_len, c_s]
        """

        # [batch_size, loop_len, c_s]
        loop_seq = batch["loop_sequence_onehot"]

        # [batch_size, loop_len]
        loop_mask = batch["loop_self_residues_mask"]

        batch_size, loop_maxlen, loop_depth = loop_seq.shape

        loop_upd, loop_att = self._loop_self_attention(loop_seq, loop_mask)

        loop_embd = loop_seq + loop_upd
        loop_embd = self.loop_dropout(loop_embd)
        loop_embd = self.loop_norm(loop_embd)
        loop_embd = self.loop_transition(loop_embd)

        # structure-based self-attention on the protein
        protein_T = Rigid.from_tensor_4x4(batch["protein_backbone_rigid_tensor"])

        # [batch_size, protein_len, c_s]
        protein_embd = batch["protein_sequence_onehot"]
        protein_norm_prox = self.protein_prox_norm(batch["protein_proximities"])

        _log.debug(f"protein_norm_prox has values ranging from {protein_norm_prox.min()} - {protein_norm_prox.max()}")
        _log.debug(f"protein_norm_prox has distribution {protein_norm_prox.mean()} +/- {protein_norm_prox.std()}")

        _log.debug(f"predictor: before ipa, protein_embd ranges {protein_embd.min()} - {protein_embd.max()}")

        protein_as = []
        protein_as_sd = []
        protein_as_b = []
        for _ in range(self.n_block):
            protein_embd, protein_a, protein_a_sd, protein_a_b = self.protein_ipa(protein_embd,
                                                                                  protein_norm_prox,
                                                                                  protein_T,
                                                                                  batch["protein_self_residues_mask"].float())
            protein_as.append(protein_a.clone().detach())
            protein_as_sd.append(protein_a_sd.detach())
            protein_as_b.append(protein_a_b.detach())

        # store the attention weights, for debugging
        # [batch_size, n_block, n_head, protein_len, protein_len]
        protein_as = torch.stack(protein_as).transpose(0, 1)
        protein_as_sd = torch.stack(protein_as_sd).transpose(0, 1)
        protein_as_b = torch.stack(protein_as_b).transpose(0, 1)

        _log.debug(f"predictor: after ipa, protein_embd ranges {protein_embd.min()} - {protein_embd.max()}")

        protein_embd = self.protein_norm(protein_embd)

        _log.debug(f"predictor: after norm, protein_embd ranges {protein_embd.min()} - {protein_embd.max()}")

        # cross attention and loop structure prediction
        output = self.cross(batch["loop_aatype"],
                            loop_embd,
                            batch["loop_cross_residues_mask"],
                            protein_embd,
                            batch["protein_cross_residues_mask"],
                            protein_T)

        output["loop_embd"] = loop_embd
        output["loop_init"] = loop_seq
        output["loop_self_attention"] = loop_att
        output["loop_pos_enc"] = get_relative_position_encoding(loop_mask, self.position_encoding_depth)
        output["protein_self_attention"] = protein_as
        output["protein_self_attention_sd"] = protein_as_sd
        output["protein_self_attention_b"] = protein_as_b

        # amino acid sequence: [1, 0, 2, ... ] meaning : Ala, Met, Cys
        # [batch_size, loop_len]
        output["aatype"] = batch["loop_aatype"]

        # whether the heavy atoms exists or not
        # for each loop residue
        # [batch_size, loop_len, 14] (true or false)
        output = make_atom14_masks(output)

        # adding hydrogens:
        # [batch_size, loop_len, 37, 3]
        output["final_atom_positions"] = atom14_to_atom37(output["final_positions"], output)

        # [batch_size, n_heads, loop_len, protein_len]
        #cross_att = output["cross_attention"]

        # [batch_size, loop_maxlen, c_s]
        loop_embd = output["single"]
        loop_embd = self.aff_dropout(self.aff_norm(loop_embd))

        output["aff_input"] = loop_embd

        # [batch_size, loop_maxlen]
        loop_embd = self.aff_trans(loop_embd).reshape(batch_size, loop_maxlen)

        if self.model_type == ModelType.REGRESSION:
            # [batch_size]
            output["affinity"] = self.output_linear(loop_embd).reshape(batch_size)

            if torch.any(torch.isnan(output["affinity"])):
                raise RuntimeError(f"got NaN output")

        elif self.model_type == ModelType.CLASSIFICATION:
            output["classification"] = self.output_linear(loop_embd)

            if torch.any(torch.isnan(output["classification"])):
                raise RuntimeError(f"got NaN output")

            output["class"] = torch.argmax(output["classification"], dim=1)

        return output

    def get_storage_size(self):
        total_size = 0
        for parameter in self.parameters():
            total_size += sys.getsizeof(parameter.storage().cpu())

        return total_size

