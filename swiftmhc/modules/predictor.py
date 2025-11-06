import logging
import ml_collections
import torch
from swiftmhc.models.model_types import ModelType
from ..tools.rigid import Rigid
from .cross_structure_module import CrossStructureModule
from .ipa import DebuggableInvariantPointAttention as SelfIPA
from .peptide_attention import PeptideSelfAttention


_log = logging.getLogger(__name__)


class Predictor(torch.nn.Module):
    """main module, calls all other modules"""

    def __init__(self, config: ml_collections.ConfigDict):
        super(Predictor, self).__init__()

        self.blosum = config.blosum

        # general settings
        self.peptide_maxlen = config.peptide_maxlen
        self.protein_maxlen = config.protein_maxlen

        self.n_head = config.num_heads
        self.n_ipa_repeat = config.num_protein_blocks

        # modules for self attention on peptide, updating {s_i}
        self.peptide_transform = torch.nn.ModuleList(
            [PeptideSelfAttention(config) for _ in range(config.num_peptide_blocks)]
        )

        # modules for self attention on protein, updating {s_j}
        self.protein_dist_norm = torch.nn.LayerNorm((self.protein_maxlen, self.protein_maxlen, 1))

        self.protein_ipa = SelfIPA(config)

        self.protein_norm = torch.nn.Sequential(
            torch.nn.Dropout(p=config.dropout_rate), torch.nn.LayerNorm(config.c_s)
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
        """Turns gradient on/off for BA modules."""
        for module in [self.ba_norm, self.ba_module]:
            for parameter in module.parameters():
                parameter.requires_grad = requires_grad

    def switch_structure_grad(self, requires_grad: bool):
        """Turns gradient on/off for structure modules."""
        for module in [
            self.peptide_transform,
            self.protein_dist_norm,
            self.protein_ipa,
            self.protein_norm,
            self.cross_sm,
        ]:
            for parameter in module.parameters():
                parameter.requires_grad = requires_grad

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """This method predicts peptide structure and BA.

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
            s_peptide = batch["peptide_blosum62"].clone()
        else:
            s_peptide = batch["peptide_sequence_onehot"].clone()

        # self attention on the peptide
        for peptide_encoder in self.peptide_transform:
            s_peptide, _ = peptide_encoder(s_peptide, batch["peptide_self_residues_mask"])

        # [*, protein_maxlen, c_s]
        if self.blosum:
            s_protein = batch["protein_blosum62"].clone()
        else:
            s_protein = batch["protein_sequence_onehot"].clone()

        protein_slice = batch["protein_self_residues_mask"].sum(dim=0).bool()
        protein_mask = protein_slice.view(1, -1, 1)

        s_protein = s_protein.masked_fill(~protein_mask, 0)
        z_protein = self.protein_dist_norm(batch["protein_proximities"])
        for _ in range(self.n_ipa_repeat):
            protein_upd, _ = self.protein_ipa(
                s_protein, z_protein, batch["protein_self_residues_mask"]
            )
            protein_upd = protein_upd.masked_fill(~protein_mask, 0)
            s_protein = self.protein_norm(s_protein + protein_upd)
            s_protein = s_protein.masked_fill(~protein_mask, 0)

        # structure-based self-attention on the protein
        T_protein = Rigid.from_tensor_4x4(batch["protein_backbone_rigid_tensor"])
        # cross attention and peptide structure prediction
        output = self.cross_sm(
            batch["ids"],
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
            batch_size = s_peptide.shape[0]
            output["affinity"] = ba.reshape(batch_size)

        elif self.model_type == ModelType.CLASSIFICATION:
            # [*, 2]
            output["logits"] = ba

            # [*]
            output["class"] = torch.argmax(ba, dim=-1)

        return output

    def _compute_ba(self, peptide_embd: torch.Tensor, peptide_mask: torch.Tensor) -> torch.Tensor:
        """Args:
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
        return ba
