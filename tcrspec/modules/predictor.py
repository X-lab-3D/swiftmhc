from typing import Dict, Union, Tuple
from copy import deepcopy as copy
import logging
import sys
from math import sqrt

from torch.nn import Embedding
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
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
from .position_encoding import get_relative_position_encoding, PositionalEncoding
from .cross_structure_module import CrossStructureModule
from ..domain.amino_acid import AMINO_ACID_DIMENSION
from ..models.data import TensorDict
from ..tools.amino_acid import one_hot_decode_sequence
from .transform import DebuggableTransformerEncoderLayer
from .ipa import DebuggableInvariantPointAttention as IPA
from ..models.types import ModelType


_log = logging.getLogger(__name__)


class Predictor(torch.nn.Module):
    def __init__(self,
                 loop_maxlen: int,
                 protein_maxlen: int,
                 model_type: ModelType,
                 config: ml_collections.ConfigDict):
        super(Predictor, self).__init__()

        structure_module_config = copy(config.structure_module)
        structure_module_config.c_s = 32
        structure_module_config.c_z = 1
        structure_module_config.no_blocks = 2
        structure_module_config.no_heads_ipa = 2
        structure_module_config.no_transition_layers = 4

        self.loop_maxlen = loop_maxlen
        self.protein_maxlen = protein_maxlen

        self.n_head = structure_module_config.no_heads_ipa

        self.posenc = PositionalEncoding(structure_module_config.c_s, self.loop_maxlen)

        self.transform = torch.nn.ModuleList([
            DebuggableTransformerEncoderLayer(structure_module_config.c_s, self.n_head)
            for _ in range(structure_module_config.no_blocks)
        ])

        self.n_ipa_repeat = structure_module_config.no_blocks

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

        c_affinity = 128

        self.affinity_reswise_mlp = torch.nn.Sequential(
            torch.nn.Linear(structure_module_config.c_s, c_affinity),
            torch.nn.GELU(),
            torch.nn.Linear(c_affinity, c_affinity),
            torch.nn.GELU(),
            torch.nn.Linear(c_affinity, 1),
        )

        self.model_type = model_type

        if model_type == ModelType.REGRESSION:
            output_size = 1
        elif model_type == ModelType.CLASSIFICATION:
            output_size = 2
        else:
            raise TypeError(str(model_type))

        self.output_linear = torch.nn.Linear(self.loop_maxlen, output_size, bias=False)

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

        # encode the loop positions
        loop_embd = self.posenc(loop_seq)

        # transform the loop
        for encoder in self.transform:
            loop_embd, loop_att = encoder(loop_embd, batch["loop_self_residues_mask"])

        # structure-based self-attention on the protein
        protein_T = Rigid.from_tensor_4x4(batch["protein_backbone_rigid_tensor"])
        protein_z = batch["protein_proximities"]
        protein_embd = batch["protein_sequence_onehot"]
        protein_mask = batch["protein_self_residues_mask"]

        for _ in range(self.n_ipa_repeat):
            protein_upd, protein_a, protein_a_sd, protein_a_b = self.protein_ipa(protein_embd,
                                                                                 protein_z,
                                                                                 protein_T,
                                                                                 protein_mask.float())
            protein_embd += protein_upd

        protein_embd = self.protein_norm(protein_embd)

        # cross attention and loop structure prediction
        output = self.cross(batch["loop_aatype"],
                            loop_embd,
                            batch["loop_cross_residues_mask"],
                            protein_embd,
                            batch["protein_cross_residues_mask"],
                            protein_T)

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

        # [batch_size, loop_len]
        p = self.affinity_reswise_mlp(loop_embd + output["single"])[..., 0]

        # affinity prediction
        if self.model_type == ModelType.REGRESSION:
            # [batch_size]
            output["affinity"] = self.output_linear(loop_embd).reshape(batch_size)

        elif self.model_type == ModelType.CLASSIFICATION:
            # softmax is required here, so that we can calculate ROC AUC
            # [batch_size, 2]
            output["classification"] = torch.nn.functional.softmax(self.affinity_linear(p), dim=1)
            # [batch_size]
            output["class"] = torch.argmax(output["classification"], dim=1)

        return output

    def get_storage_size(self):
        total_size = 0
        for parameter in self.parameters():
            total_size += sys.getsizeof(parameter.storage().cpu())

        return total_size

