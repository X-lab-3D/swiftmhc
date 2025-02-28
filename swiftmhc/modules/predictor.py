from typing import Dict, Union
from copy import deepcopy as copy
import logging
import sys
from math import sqrt

from torch.nn import Embedding
from torch.nn.modules.transformer import TransformerEncoder
import torch
import ml_collections

from openfold.utils.loss import compute_plddt
from openfold.data.data_transforms import make_atom14_masks
from openfold.data.data_pipeline import make_sequence_features
from openfold.utils.loss import find_structural_violations
from openfold.utils.feats import atom14_to_atom37
from openfold.model.primitives import LayerNorm
from openfold.np.residue_constants import restypes

from ..tools.rigid import Rigid
from ..domain.amino_acid import AMINO_ACID_DIMENSION
from ..models.data import TensorDict
from ..tools.amino_acid import one_hot_decode_sequence
from .ipa import DebuggableInvariantPointAttention as SelfIPA
from .peptide_attention import PeptideSelfAttention
from .cross_structure_module import CrossStructureModule
from ..models.types import ModelType


_log = logging.getLogger(__name__)


class Predictor(torch.nn.Module):
    """
    main module, calls all other modules
    """

    def __init__(self, config: ml_collections.ConfigDict):
        super(Predictor, self).__init__()

        self.blosum = config.blosum

        # general settings
        self.peptide_maxlen = config.peptide_maxlen
        self.protein_maxlen = config.protein_maxlen

        self.n_head = config.no_heads
        self.n_ipa_repeat = config.no_protein_blocks

        # modules for self attention on peptide, updating {s_i}
        self.peptide_transform = torch.nn.ModuleList([
            PeptideSelfAttention(config)
            for _ in range(config.no_peptide_blocks)
        ])

        self.peptide_norm = torch.nn.Sequential(
            torch.nn.Dropout(p=config.dropout_rate),
            torch.nn.LayerNorm(config.c_s)
        )

        self.protein_ipa = SelfIPA(config)

        self.protein_norm = torch.nn.Sequential(
            torch.nn.Dropout(p=config.dropout_rate),
            torch.nn.LayerNorm(config.c_s)
        )

        # module for structure prediction and cross updating {s_i}
        self.cross_sm = CrossStructureModule(config)

        # decide here what output shape to generate: regression or classification
        self.model_type = config.model_type
        if self.model_type == ModelType.REGRESSION:
            output_size = 1
        elif self.model_type == ModelType.CLASSIFICATION:
            output_size = 2
        else:
            raise TypeError(str(self.model_type))

        self.ba_norm = torch.nn.Sequential(
            torch.nn.Dropout(p=config.dropout_rate),
            torch.nn.LayerNorm(config.c_s),
        )

        # module for predicting affinity from updated {s_i}
        self.ba_module = torch.nn.Sequential(
            torch.nn.Linear(config.c_s, config.c_transition),
            torch.nn.ReLU(),
            torch.nn.Linear(config.c_transition, output_size),
        )

    def switch_affinity_grad(self, requires_grad: bool):
        """
        Turns gradient on/off for BA modules.
        """

        for module in [self.ba_norm, self.ba_module]:
            for parameter in module.parameters():
                parameter.requires_grad = requires_grad

    def switch_structure_grad(self, requires_grad: bool):
        """
        Turns gradient on/off for structure modules.
        """

        for module in [self.peptide_transform, self.peptide_norm,
                       self.protein_ipa,
                       self.protein_norm, self.cross_sm]:

            for parameter in module.parameters():
                parameter.requires_grad = requires_grad

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        This method predicts peptide structure and BA.

        Args:
            peptide_aatype:                 [*, peptide_maxlen] (int, 0 - 19, sequence)
            peptide_sequence_onehot:        [*, peptide_maxlen, c_s] (encoded sequence)
            peptide_blosum62:               [*, peptide_maxlen, c_s] (encoded sequence)
            peptide_self_residues_mask:     [*, peptide_maxlen] (bool, residue used or not)
            peptide_cross_residue_mask:     [*, peptide_maxlen] (bool, residue used or not)
            protein_sequence_onehot:        [*, protein_maxlen, c_s] (encoded sequence)
            protein_blosum62:               [*, protein_maxlen, c_s] (encoded sequence)
            protein_self_residues_mask:     [*, protein_maxlen] (bool, residue used or not)
            protein_cross_residues_mask:    [*, protein_maxlen] (bool, residue used or not)
            protein_backbone_rigid_tensor:  [*, protein_maxlen, 4, 4] (converted to frames, backbone only)
            protein_proximities:            [*, protein_maxlen, protein_maxlen, 1] (inverted distances)

        Returns:
            single:                     [*, peptide_maxlen, c_s] (updated {s_i})
            final_frames:               [*, peptide_maxlen, 7] (converted from frames, backbone only)
            final_sidechain_frames:     [*, peptide_maxlen, 7, 4, 4] (converted from frames, sidechain)
            final_angles:               [*, peptide_maxlen, 7, 2] (sin,cos)
            final_unnormalized_angles:  [*, peptide_maxlen, 7, 2] (sin,cos but not normalized)
            final_positions:            [*, peptide_maxlen, 14, 3] (position of each atom in 14-atom format)
            affinity:                   [*] (BA, for regression only)
            logits:                     [*, 2] (BA, for classification only)
            class:                      [*] (BA, binary 0 / 1, for classification only)
        """

        # [*, peptide_maxlen, c_s]
        if self.blosum:
            s_peptide = batch["peptide_blosum62"]
        else:
            s_peptide = batch["peptide_sequence_onehot"]

        batch_size, peptide_maxlen, peptide_dim = s_peptide.shape

        # Ignore residues that are masked all over the batch.
        peptide_slice = batch["peptide_self_residues_mask"].sum(dim=0).bool()
        protein_slice = batch["protein_self_residues_mask"].sum(dim=0).bool()
        protein_slice_length = protein_slice.sum()
        protein_2d_slice = torch.logical_and(protein_slice[None, :], protein_slice[:, None])

        # self attention on the peptide
        for peptide_encoder in self.peptide_transform:
            peptide_upd, a = peptide_encoder(
                s_peptide[:, peptide_slice],
                batch["peptide_self_residues_mask"][:, peptide_slice],
            )
            s_peptide[:, peptide_slice] = self.peptide_norm(s_peptide[:, peptide_slice] + peptide_upd)

        # structure-based self-attention on the protein
        T_protein = Rigid.from_tensor_4x4(batch["protein_backbone_rigid_tensor"])

        # [*, protein_maxlen, c_s]
        if self.blosum:
            s_protein = batch["protein_blosum62"]
        else:
            s_protein = batch["protein_sequence_onehot"]

        z_protein = batch["protein_proximities"]

        # normalize
        z_protein = (z_protein - z_protein.mean(dim=(-3, -2, -1))[..., None, None, None]) / (1e-6 + z_protein.std(dim=(-3, -2, -1))[..., None, None, None])

        sliced_z_protein = z_protein[:, protein_2d_slice].reshape(batch_size, protein_slice_length, protein_slice_length, -1)

        for _ in range(self.n_ipa_repeat):
            protein_upd, a = self.protein_ipa(
                s_protein[:, protein_slice],
                sliced_z_protein,
                batch["protein_self_residues_mask"][:, protein_slice].to(dtype=s_protein.dtype),
            )
            s_protein[:, protein_slice] = self.protein_norm(s_protein[:, protein_slice] + protein_upd)

        # cross attention and peptide structure prediction
        output = self.cross_sm(
            batch["peptide_aatype"],
            s_peptide,
            batch["peptide_cross_residues_mask"],
            s_protein,
            batch["protein_cross_residues_mask"],
            T_protein,
        )

        # retrieve updated peptide sequence, per residue features
        # normalization and skip connection
        # [*, peptide_maxlen, c_s]
        s_peptide = self.ba_norm(s_peptide + output["single"])

        # [*, peptide_maxlen]
        peptide_mask = batch["peptide_cross_residues_mask"]

        # [*, output_size]
        ba = self._compute_ba(s_peptide, peptide_mask)

        # affinity prediction
        if self.model_type == ModelType.REGRESSION:

            # reshape a 1-dimensional vector into a scalar
            # [*]
            output["affinity"] = ba.reshape(batch_size)

        elif self.model_type == ModelType.CLASSIFICATION:

            # [*, 2]
            output["logits"] = ba

            # [*]
            output["class"] = torch.argmax(ba, dim=-1)

        return output

    def _compute_ba(self, peptide_embd: torch.Tensor, peptide_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            peptide_embd:       [*, peptide_maxlen, c_s]
            peptide_mask:       [*, peptide_maxlen]
        Return:
            ba:                 [*, output_size(1 or 2)]
        """

        # [*, peptide_maxlen, output_size]
        peptide_scores = self.ba_module(peptide_embd)

        # [*, peptide_maxlen, output_size]
        masked_scores = torch.where(peptide_mask.unsqueeze(-1), peptide_scores, 0.0)

        # [*, output_size]
        ba = masked_scores.sum(dim=-2)

        return ba

