from typing import Dict, Union
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

from .position_encoding import RelativePositionEncoder
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
                 peptide_maxlen: int,
                 protein_maxlen: int,
                 model_type: ModelType,
                 config: ml_collections.ConfigDict):
        super(Predictor, self).__init__()

        structure_module_config = copy(config.structure_module)
        structure_module_config.c_s = 32
        structure_module_config.c_z = 1
        structure_module_config.no_blocks = 2
        structure_module_config.no_heads_ipa = 2

        self.peptide_maxlen = peptide_maxlen
        self.protein_maxlen = protein_maxlen

        self.n_head = structure_module_config.no_heads_ipa

        self.transform = torch.nn.ModuleList([
            RelativePositionEncoder(self.n_head, self.peptide_maxlen, structure_module_config.c_s)
            for _ in range(structure_module_config.no_blocks)
        ])

        self.peptide_norm = torch.nn.Sequential(
            torch.nn.Dropout(p=structure_module_config.dropout_rate),
            LayerNorm(structure_module_config.c_s)
        )

        self.n_ipa_repeat = structure_module_config.no_blocks

        self.protein_dist_norm = torch.nn.LayerNorm((self.protein_maxlen, self.protein_maxlen, 1))

        self.protein_ipa = IPA(
            structure_module_config.c_s,
            structure_module_config.c_z,
            structure_module_config.c_ipa,
            structure_module_config.no_heads_ipa,
            structure_module_config.no_qk_points,
            structure_module_config.no_v_points
        )
        self.protein_ipa.inf = structure_module_config.inf

        self.protein_norm = torch.nn.Sequential(
            torch.nn.Dropout(p=structure_module_config.dropout_rate),
            LayerNorm(structure_module_config.c_s)
        )

        self.cross = CrossStructureModule(**structure_module_config)

        c_transition = 128
        c_interaction = 64

        self.protein_feature = torch.nn.Sequential(
            torch.nn.Linear(structure_module_config.c_s, c_interaction, bias=False),
            torch.nn.Tanh(),
        )

        self.peptide_feature = torch.nn.Sequential(
            torch.nn.Linear(structure_module_config.c_s, c_interaction, bias=False),
            torch.nn.Tanh(),
        )

        self.head_weight = torch.nn.Sequential(
            torch.nn.Linear(self.n_ipa_repeat * self.n_head, 1),
            torch.nn.Softmax(-1),
        )

        self.model_type = model_type

        if model_type == ModelType.REGRESSION:
            output_size = 1
        elif model_type == ModelType.CLASSIFICATION:
            output_size = 2
        else:
            raise TypeError(str(model_type))

        self.affinity_linear = torch.nn.Linear(c_interaction, output_size)

    def predict_ba(
        self,
        a: torch.Tensor,
        s_peptide: torch.Tensor,
        s_protein: torch.Tensor,
        mask_peptide: torch.Tensor,
        mask_protein: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, peptide_maxlen, _ = s_peptide.shape
        protein_maxlen = s_protein.shape[1]

        # [*, peptide_maxlen, c] ranges (-1 - 1)
        f_peptide = self.peptide_feature(s_peptide) * mask_peptide.unsqueeze(-1)

        # [*, protein_maxlen, c] ranges (-1 - 1)
        f_protein = self.protein_feature(s_protein) * mask_protein.unsqueeze(-1)

        # [*]
        nres_peptide = mask_peptide.int().sum(dim=-1)

        # [*]
        nres_protein = mask_protein.int().sum(dim=-1)

        c = f_peptide.shape[-1]

        # [*, peptide_maxlen, protein_maxlen, c]
        ff = f_peptide.unsqueeze(-2) * f_protein.unsqueeze(-3)

        # [*, n_block * n_head, peptide_maxlen, protein_maxlen]
        w = a.reshape(batch_size, -1, peptide_maxlen, protein_maxlen)

        # [*, peptide_maxlen, protein_maxlen, n_block * n_head]
        w = w.transpose(-3, -2).transpose(-2, -1)

        # [*, peptide_maxlen, protein_maxlen, 1]
        w = self.head_weight(w)

        # [*, peptide_maxlen, protein_maxlen, c]
        wff = w * ff / sqrt(c)

        # [*, output_size]
        p = self.affinity_linear(wff).sum(dim=(-3, -2))

        # take mean
        p = p / (nres_peptide * nres_protein).unsqueeze(-1)

        return p


    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            peptide_aatype:                 [*, peptide_maxlen] (int, 0 - 19)
            peptide_sequence_onehot:        [*, peptide_maxlen, c_s]
            peptide_self_residues_mask:     [*, peptide_maxlen]
            peptide_cross_residue_mask:     [*, peptide_maxlen]
            protein_sequence_onehot:        [*, protein_maxlen, c_s]
            protein_self_residues_mask:     [*, protein_maxlen]
            protein_cross_residues_mask:    [*, protein_maxlen]
            protein_backbone_rigid_tensor:  [*, protein_maxlen, 4, 4]
            protein_proximities:            [*, protein_maxlen, protein_maxlen, 1]

        Returns:
            single:                     [*, peptide_maxlen, c_s]
            final_frames:               [*, peptide_maxlen, 7]
            final_sidechain_frames:     [*, peptide_maxlen, 7, 4, 4]
            final_angles:               [*, peptide_maxlen, 7, 2] (cos(a), sin(a))
            final_unnormalized_angles:  [*, peptide_maxlen, 7, 2] (cos(a), sin(a))
            final_positions:            [*, peptide_maxlen, 14, 3]
            atom14_atom_exists:         [*, peptide_maxlen, 14]
            final_atom_positions:       [*, peptide_maxlen, 37, 3]
            atom37_atom_exists:         [*, peptide_maxlen, 37]
            affinity:                   [*] (for regression only)
            classification:             [*, 2] (for classification only)
            class:                      [*] (0 / 1, for classification only)
        """

        # [*, peptide_maxlen, c_s]
        peptide_seq = batch["peptide_sequence_onehot"]
        batch_size, peptide_maxlen, peptide_depth = peptide_seq.shape

        # transform the peptide
        peptide_embd = peptide_seq.clone()
        for encoder in self.transform:
            peptide_upd, peptide_att = encoder(peptide_embd, batch["peptide_self_residues_mask"])
            peptide_embd = self.peptide_norm(peptide_embd + peptide_upd)

        # structure-based self-attention on the protein
        protein_T = Rigid.from_tensor_4x4(batch["protein_backbone_rigid_tensor"])

        # [*, protein_maxlen, c_s]
        protein_embd = batch["protein_sequence_onehot"]
        protein_norm_prox = self.protein_dist_norm(batch["protein_proximities"])

        protein_as = []
        for _ in range(self.n_ipa_repeat):
            protein_upd, protein_a = self.protein_ipa(protein_embd,
                                                      protein_norm_prox,
                                                      protein_T,
                                                      batch["protein_self_residues_mask"].float())
            protein_as.append(protein_a.clone().detach())
            protein_embd =  self.protein_norm(protein_embd + protein_upd)

        # store the attention weights, for debugging
        # [*, n_block, H, protein_maxlen, protein_maxlen]
        protein_as = torch.stack(protein_as).transpose(0, 1)

        # cross attention and peptide structure prediction
        output = self.cross(batch["peptide_aatype"],
                            peptide_embd,
                            batch["peptide_cross_residues_mask"],
                            protein_embd,
                            batch["protein_cross_residues_mask"],
                            protein_T)

        output["protein_self_attention"] = protein_as

        # amino acid sequence: [1, 0, 2, ... ] meaning : Ala, Met, Cys
        # [*, peptide_maxlen]
        output["aatype"] = batch["peptide_aatype"]

        # whether the heavy atoms exists or not
        # for each peptide residue
        output = make_atom14_masks(output)

        # converting 14-atom format to 37-atom format:
        # [*, peptide_maxlen, 37, 3]
        output["final_atom_positions"] = atom14_to_atom37(output["final_positions"], output)

        peptide_embd = output["single"]

        # [*, output_size]
        ba_output = self.predict_ba(
            output["cross_ipa_att"],
            peptide_embd,
            protein_embd,
            batch["peptide_cross_residues_mask"],
            batch["protein_cross_residues_mask"],
        )

        # affinity prediction
        if self.model_type == ModelType.REGRESSION:
            # [*]
            output["affinity"] = ba_output.reshape(batch_size)

        elif self.model_type == ModelType.CLASSIFICATION:

            # [*, 2]
            output["logits"] = ba_output

            # [*]
            output["class"] = torch.argmax(ba_output, dim=1)

        return output

    def get_storage_size(self):
        total_size = 0
        for parameter in self.parameters():
            total_size += sys.getsizeof(parameter.storage().cpu())

        return total_size
