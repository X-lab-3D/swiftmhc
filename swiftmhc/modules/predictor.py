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

from position_encoding.relative import get_relative_position_encoding_matrix

from ..tools.rigid import Rigid
from .cross_structure_module import CrossStructureModule
from ..domain.amino_acid import AMINO_ACID_DIMENSION
from ..models.data import TensorDict
from ..tools.amino_acid import one_hot_decode_sequence
from .ipa import DebuggableInvariantPointAttention as IPA
from .cross_ipa import CrossInvariantPointAttention as CrossIPA
from ..models.types import ModelType


_log = logging.getLogger(__name__)


class Predictor(torch.nn.Module):
    """
    main module, calls all other modules
    """

    def __init__(self,
        peptide_maxlen: int,
        protein_maxlen: int,
        model_type: ModelType,
        config: ml_collections.ConfigDict,
        blosum: bool,
    ):
        super(Predictor, self).__init__()

        self.blosum = blosum

        # general settings
        self.eps = 1e-6

        structure_module_config = copy(config.structure_module)
        structure_module_config.c_s = 32
        structure_module_config.c_z = 1
        structure_module_config.no_blocks = 2
        structure_module_config.no_heads_ipa = 2

        self.peptide_maxlen = peptide_maxlen
        self.protein_maxlen = protein_maxlen

        self.n_head = structure_module_config.no_heads_ipa
        self.n_ipa_repeat = structure_module_config.no_blocks

        # modules for self attention on peptide, updating {s_i}
        self.relpos_encode = torch.nn.Linear(self.peptide_maxlen * (self.peptide_maxlen * 2 - 1), structure_module_config.c_s)

        # modules for self attention on protein, updating {s_j}
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

        # module for structure prediction and cross updating {s_i}
        self.cross = CrossStructureModule(**structure_module_config)

        # decide here what output shape to generate: regression or classification
        self.model_type = model_type
        if model_type == ModelType.REGRESSION:
            output_size = 1
        elif model_type == ModelType.CLASSIFICATION:
            output_size = 2
        else:
            raise TypeError(str(model_type))

        # module for predicting affinity from updated {s_i}
        # residue-wise mlp
        c_transition = 128
        self.affinity_module = torch.nn.Sequential(
            torch.nn.Linear(structure_module_config.c_s, c_transition),
            torch.nn.ReLU(),
            torch.nn.Linear(c_transition, output_size),
        )

    def _encode_peptide(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [*, N, c_s] amino acid representation of peptide
            mask: [*, N] (bool) which peptide residues exist or not
        Returns:
            [*, N, c_s] position encoded representation of peptide
        """

        # [N, N, 2 * N - 1]
        relpos = get_relative_position_encoding_matrix(mask.shape[-1])

        # [N, N]
        square_mask = mask[..., :, None] * mask[..., None, :]

        # set nonexistent neighbours to zero
        # [*, N, N, 2 * N - 1]
        relpos = relpos[None, ...] * square_mask[..., None]

        # [*, N, c_s]
        b = self.relpos_encode(relpos.reshape(list(relpos.shape[:-2]) + [-1]).float())

        return b + features


    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            peptide_aatype:                 [*, peptide_maxlen] (int, 0 - 19)
            peptide_sequence_onehot:        [*, peptide_maxlen, c_s]
            peptide_blosum62:               [*, peptide_maxlen, c_s]
            peptide_self_residues_mask:     [*, peptide_maxlen]
            peptide_cross_residue_mask:     [*, peptide_maxlen]
            protein_sequence_onehot:        [*, protein_maxlen, c_s]
            protein_blosum62:               [*, protein_maxlen, c_s]
            protein_self_residues_mask:     [*, protein_maxlen]
            protein_cross_residues_mask:    [*, protein_maxlen]
            protein_backbone_rigid_tensor:  [*, protein_maxlen, 4, 4]
            protein_proximities:            [*, protein_maxlen, protein_maxlen, 1]

        Returns:
            single:                     [*, peptide_maxlen, c_s] (updated {s_i})
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
        if self.blosum:
            peptide_seq = batch["peptide_blosum62"]
        else:
            peptide_seq = batch["peptide_sequence_onehot"]

        batch_size, peptide_maxlen, peptide_depth = peptide_seq.shape

        # Ignore residues that are masked all over the batch.
        peptide_slice = batch["peptide_self_residues_mask"].sum(dim=0).bool()
        protein_slice = batch["protein_self_residues_mask"].sum(dim=0).bool()
        protein_slice_length = protein_slice.sum()
        protein_prox_slice = torch.logical_and(protein_slice[None, :], protein_slice[:, None])

        # self attention on the peptide
        peptide_embd = self._encode_peptide(peptide_seq.clone(), batch["peptide_self_residues_mask"])

        # structure-based self-attention on the protein
        protein_T = Rigid.from_tensor_4x4(batch["protein_backbone_rigid_tensor"])

        # [*, protein_maxlen, c_s]
        if self.blosum:
            protein_embd = batch["protein_blosum62"].clone()
        else:
            protein_embd = batch["protein_sequence_onehot"].clone()
        protein_norm_prox = self.protein_dist_norm(batch["protein_proximities"])
        sliced_protein_norm_prox = protein_norm_prox[:, protein_prox_slice].reshape(batch_size, protein_slice_length, protein_slice_length, -1)

        for _ in range(self.n_ipa_repeat):
            protein_upd, protein_a = self.protein_ipa(protein_embd[:, protein_slice],
                                                      sliced_protein_norm_prox,
                                                      protein_T[:, protein_slice],
                                                      batch["protein_self_residues_mask"][:, protein_slice].float())
            protein_embd[:, protein_slice] = self.protein_norm(protein_embd[:, protein_slice] + protein_upd)

        # cross attention and peptide structure prediction
        output = self.cross(batch["peptide_aatype"],
                            peptide_embd,
                            batch["peptide_cross_residues_mask"],
                            protein_embd,
                            batch["protein_cross_residues_mask"],
                            protein_T)

        # amino acid sequence: [1, 0, 2, ... ] meaning : Arg, Ala, Asn
        # [*, peptide_maxlen]
        output["aatype"] = batch["peptide_aatype"]

        # whether the heavy atoms exists or not
        # for each peptide residue
        output = make_atom14_masks(output)

        # converting 14-atom format to 37-atom format:
        # [*, peptide_maxlen, 37, 3]
        output["final_atom_positions"] = atom14_to_atom37(output["final_positions"], output)

        peptide_embd = output["single"]

        # [*, peptide_maxlen, output_size]
        p = self.affinity_module(peptide_embd)

        # [*, peptide_maxlen, 1]
        mask = batch["peptide_cross_residues_mask"].unsqueeze(-1)

        # [*]
        length = batch["peptide_cross_residues_mask"].float().sum(dim=-1)

        # [*, peptide_maxlen, output_size]
        masked_p = torch.where(mask, p, 0.0)

        # [*, output_size]
        ba_output = masked_p.sum(dim=-2)

        # affinity prediction
        if self.model_type == ModelType.REGRESSION:
            # [*]
            output["affinity"] = ba_output.reshape(batch_size)

        elif self.model_type == ModelType.CLASSIFICATION:

            # [*, 2]
            output["logits"] = ba_output

            # [*]
            output["class"] = torch.argmax(ba_output, dim=-1)

        return output

