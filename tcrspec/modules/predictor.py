from typing import Dict, Union
from copy import deepcopy as copy
import logging

from torch.nn import Embedding
from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder
import torch
import ml_collections

from openfold.model.structure_module import InvariantPointAttention as IPA
from openfold.utils.rigid_utils import Rigid, Rotation
from openfold.utils.loss import compute_plddt
from openfold.data.data_transforms import make_atom14_masks
from openfold.data.data_pipeline import make_sequence_features
from openfold.utils.loss import find_structural_violations
from openfold.utils.feats import atom14_to_atom37
from openfold.model.primitives import LayerNorm

from .position_encoding import PositionalEncoding
from .cross_structure_module import CrossStructureModule
from ..domain.amino_acid import AMINO_ACID_DIMENSION
from ..models.data import TensorDict


_log = logging.getLogger(__name__)


class Predictor(torch.nn.Module):
    def __init__(self,
                 config: ml_collections.ConfigDict):
        super(Predictor, self).__init__()

        structure_module_config = copy(config.structure_module)
        structure_module_config.c_s = 32
        structure_module_config.c_z = 1
        structure_module_config.no_blocks = 2
        structure_module_config.no_heads_ipa = 2

        self.loop_maxlen = 16
        self.protein_maxlen = 40

        self.pos_enc = PositionalEncoding(structure_module_config.c_s, self.loop_maxlen)

        self.loop_enc = TransformerEncoder(
            TransformerEncoderLayer(structure_module_config.c_s,
                                    structure_module_config.no_heads_ipa,
                                    batch_first=True),
            structure_module_config.no_blocks
        )

        self.n_ipa_repeat = structure_module_config.no_blocks

        self.protein_ipa = IPA(structure_module_config.c_s,
                               structure_module_config.c_z,
                               structure_module_config.c_ipa,
                               structure_module_config.no_heads_ipa,
                               structure_module_config.no_qk_points,
                               structure_module_config.no_v_points)

        self.protein_norm = torch.nn.Sequential(
            LayerNorm(structure_module_config.c_s),
            torch.nn.Dropout(p=0.1)
        )

        self.cross = CrossStructureModule(**structure_module_config)

        c_affinity = 200

        #self.aff_norm = LayerNorm(structure_module_config.c_s)

        self.aff_trans = torch.nn.Sequential(
            torch.nn.Linear(structure_module_config.c_s, 10),
            torch.nn.GELU(),
            torch.nn.Linear(10, structure_module_config.c_s),
            torch.nn.Dropout(0.1),
            torch.nn.LayerNorm(structure_module_config.c_s)
        )

        mlp_input_size = self.loop_maxlen * structure_module_config.c_s
        #mlp_input_size = self.loop_maxlen * self.protein_maxlen * structure_module_config.no_heads_ipa

        self.aff_mlp = torch.nn.Sequential(
            torch.nn.Linear(mlp_input_size, c_affinity),
            torch.nn.GELU(),
            torch.nn.Linear(c_affinity, c_affinity),
            torch.nn.GELU(),
            torch.nn.Linear(c_affinity, 1)
        )

    def forward(self, batch: TensorDict) -> TensorDict:
        """
            Returns:
                single:               [batch_size, loop_len, c_s]
                aatype:               [batch_size, loop_len]  (int)
                residue_index:        [batch_size, loop_len]  (int)
                affinity:             [batch_size]

                frames:               [n_blocks, batch_size, loop_len, 4, 4]
                sidechain_frames:     [n_blocks, batch_size, loop_len, 8, 4, 4]
                unnormalized_angles:  [n_blocks, batch_size, loop_len, 7, 2]
                angles:               [n_blocks, batch_size, loop_len, 7, 2]
                positions:            [n_blocks, batch_size, loop_len, 14, 3]
                states:               [n_blocks, batch_size, loop_len, c_s]
        """

        # [batch_size, loop_len, c_s]
        loop_embd = batch["loop_sequence_embedding"].clone()
        batch_size = loop_embd.shape[0]

        # positional encoding
        loop_embd = self.pos_enc(loop_embd)

        # self-attention on the loop
        loop_embd = self.loop_enc(loop_embd, src_key_padding_mask=batch["loop_len_mask"])

        # structure-based self-attention on the protein
        protein_T = Rigid.from_tensor_4x4(batch["protein_backbone_rigid_tensor"])

        # [batch_size, protein_len, c_s]
        protein_embd = batch["protein_sequence_embedding"].clone()

        for _ in range(self.n_ipa_repeat):
            protein_embd = self.protein_ipa(protein_embd,
                                            batch["protein_proximities"],
                                            protein_T,
                                            batch["protein_len_mask"].float())

        protein_embd = self.protein_norm(protein_embd)

        # cross attention and loop structure prediction
        output = self.cross(batch["loop_aatype"],
                            loop_embd,
                            batch["loop_len_mask"],
                            protein_embd,
                            batch["protein_len_mask"],
                            protein_T)

        # amino acid sequence: [1, 0, 2, ... ] meaning : Ala, Met, Cys
        # [batch_size, loop_len]
        output["aatype"] = batch["loop_aatype"]

        # amino acid sequence index: [0, 1, 2, 3, 4, ... ], representing the order of amino acids
        # [batch_size, loop_len]
        output["residue_index"] = torch.arange(0,
                                               batch["loop_sequence_embedding"].shape[1], 1,
                                               dtype=torch.int64,
                                               device=batch["loop_sequence_embedding"].device
        ).unsqueeze(dim=0).expand(batch_size, -1)

        # whether the heavy atoms exists or not
        # for each loop residue
        # [batch_size, loop_len, 14] (true or false)
        output = make_atom14_masks(output)

        # adding hydrogens:
        # [batch_size, loop_len, 37, 3]
        output["final_atom_positions"] = atom14_to_atom37(output["final_positions"], output)

        # [batch_size, n_heads, loop_len, protein_len]
        cross_att = output["cross_attention"]

        # transition on s_loop before prediction BA
        updated_s_loop = self.aff_trans(output["single"])

        # [batch_size, loop_maxlen]
        #output["affinity"] = self.aff_mlp(cross_att.reshape(batch_size, -1)).reshape(batch_size)
        output["affinity"] = self.aff_mlp(updated_s_loop.reshape(batch_size, -1)).reshape(batch_size)

        return output





