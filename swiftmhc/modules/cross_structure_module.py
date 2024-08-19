from typing import Dict, Union, Optional
import logging

import torch

from openfold.utils.rigid_utils import Rotation
from openfold.model.primitives import Linear, LayerNorm
from openfold.model.structure_module import AngleResnet, StructureModuleTransition
from openfold.utils.tensor_utils import dict_multimap
from openfold.utils.feats import (
    frames_and_literature_positions_to_atom14_pos,
    torsion_angles_to_frames,
)
from openfold.np.residue_constants import (
    restype_rigid_group_default_frame,
    restype_atom14_to_rigid_group,
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
)

from ..tools.rigid import Rigid
from .cross_ipa import CrossInvariantPointAttention
from ..operate import average_rigid


_log = logging.getLogger(__name__)


class BackboneUpdate(torch.nn.Module):
    """
    Implements part of Algorithm 23.
    """

    def __init__(self, c_s):
        """
        Args:
            c_s:
                Single representation channel dimension
        """
        super(BackboneUpdate, self).__init__()

        self.c_s = c_s

        self.c_transition = 128

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.c_s, self.c_transition),
            torch.nn.ReLU(),
            Linear(self.c_transition, 6, init="final")
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            [*, N_res, C_s] single representation
        Returns:
            [*, N_res, 6] update vector 
        """
        # [*, 6]
        update = self.mlp(s)

        return update


class CrossStructureModule(torch.nn.Module):
    """
    This is like algorithm 20 in AlphaFold2, but with some modifications:

     - omega angles are calculated from predicted frames. They are not predicted directly.
     - The backbone frames are predicted from s_i, but not by a single linear layer, rather a transitional block.
     - It does not predict frames for a complete sequence. It takes a protein structure and peptide sequence as input. The peptide structure is predicted.

    The code was copied from OpenFold and then modified.
    """

    def __init__(
        self,
        c_s,
        c_ipa,
        c_resnet,
        no_heads_ipa,
        no_qk_points,
        no_v_points,
        dropout_rate,
        no_blocks,
        no_resnet_blocks,
        no_angles,
        trans_scale_factor,
        no_transition_layers,
        epsilon,
        **kwargs,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_ipa:
                IPA hidden channel dimension
            c_resnet:
                Angle resnet (Alg. 23 lines 11-14) hidden channel dimension
            no_heads_ipa:
                Number of IPA heads
            no_qk_points:
                Number of query/key points to generate during IPA
            no_v_points:
                Number of value points to generate during IPA
            dropout_rate:
                Dropout rate used throughout the layer
            no_blocks:
                Number of structure module blocks
            no_resnet_blocks:
                Number of blocks in the angle resnet
            no_angles:
                Number of angles to generate in the angle resnet
            no_transition_layers:
                Number of layers to use for transition
            trans_scale_factor:
                Scale of single representation transition hidden dimension
            epsilon:
                Small number used in angle resnet normalization
        """
        super(CrossStructureModule, self).__init__()

        # constants
        self.c_s = c_s
        self.c_ipa = c_ipa
        self.c_resnet = c_resnet
        self.no_heads_ipa = no_heads_ipa
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.dropout_rate = dropout_rate
        self.n_blocks = no_blocks
        self.no_resnet_blocks = no_resnet_blocks
        self.no_angles = no_angles
        self.trans_scale_factor = trans_scale_factor
        self.epsilon = epsilon
        self.n_transition_layers = no_transition_layers

        # Buffers to be lazily initialized later
        # self.default_frames
        # self.group_idx
        # self.atom_mask
        # self.lit_positions

        # initial modules, to run on s_i
        self.layer_norm_s_peptide = LayerNorm(self.c_s)
        self.layer_norm_s_protein = LayerNorm(self.c_s)

        self.linear_in_peptide = Linear(self.c_s, self.c_s)
        self.linear_in_protein = Linear(self.c_s, self.c_s)

        # modules for updating s_i (peptide), from the protein structure
        self.peptide_ipa = CrossInvariantPointAttention(
            self.c_s,
            self.c_ipa,
            self.no_heads_ipa,
            self.no_qk_points,
            self.no_v_points,
            eps=self.epsilon,
        )
        self.peptide_ipa_dropout = torch.nn.Dropout(self.dropout_rate)
        self.peptide_layer_norm_ipa = LayerNorm(self.c_s)
        self.peptide_transition = StructureModuleTransition(self.c_s,
                                                            self.n_transition_layers,
                                                            self.dropout_rate)

        # for predicting backbone frames from s_i
        self.bb_update = BackboneUpdate(self.c_s)

        # for predicting torsion angles
        self.angle_resnet = AngleResnet(
            self.c_s,
            self.c_resnet,
            self.no_resnet_blocks,
            self.no_angles,
            self.epsilon,
        )

    def forward(
        self,
        peptide_aatype: torch.Tensor,
        s_peptide_initial: torch.Tensor,
        peptide_mask: torch.Tensor,
        s_protein_initial: torch.Tensor,
        protein_mask: torch.Tensor,
        T_protein: Rigid

    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            peptide_aatype:         [*, peptide_maxlen] (0 - 19)
            s_peptide_initial:      [*, peptide_maxlen, c_s]
            peptide_mask:           [*, peptide_maxlen]
            s_protein_initial:      [*, protein_maxlen, c_s]
            protein_mask:           [*, protein_maxlen]
            T_protein:              [*, protein_maxlen, 4, 4]
        Returns:
            frames:                 [*, peptide_maxlen, 4, 4]
            sidechain_frames:       [*, peptide_maxlen, 4, 4]
            unnormalized_angles:    [*, peptide_maxlen, 7, 2]
            angles:                 [*, peptide_maxlen, 7, 2]
            positions:              [*, peptide_maxlen, 18, 3]
            states:                 [*, peptide_maxlen, c_s]
            single:                 [*, peptide_maxlen, c_s]
        """

        batch_size, peptide_maxlen, embd_depth = s_peptide_initial.shape

        # Ignore residues that are masked all across the batch.
        peptide_slice = peptide_mask.sum(dim=0).bool()
        protein_slice = protein_mask.sum(dim=0).bool()

        # slice out those masked residues, for performance reasons.
        s_peptide_initial = s_peptide_initial[:, peptide_slice]
        s_protein_initial = s_protein_initial[:, protein_slice]
        T_protein = T_protein[:, protein_slice]
        protein_mask = protein_mask[:, protein_slice]
        peptide_mask = peptide_mask[:, peptide_slice]
        peptide_aatype = peptide_aatype[:, peptide_slice]

        # [*, peptide_maxlen, c_s]
        s_peptide_initial = self.layer_norm_s_peptide(s_peptide_initial)

        # [*, protein_maxlen, c_s]
        s_protein_initial = self.layer_norm_s_protein(s_protein_initial)

        # [*, peptide_maxlen, c_s]
        s_peptide = torch.clone(s_peptide_initial)
        s_peptide = self.linear_in_peptide(s_peptide)

        # [*, protein_maxlen, c_s]
        s_protein = torch.clone(s_protein_initial)
        s_protein = self.linear_in_protein(s_protein)

        # [*, peptide_maxlen]
        T_peptide = Rigid.identity(
            s_peptide.shape[:-1],
            s_peptide.dtype,
            s_peptide.device,
            self.training,
            fmt="quat",
        )

        # update s_i repeatedly
        outputs = []
        for i in range(self.n_blocks):

            preds = self._block(
                s_peptide_initial,
                peptide_aatype,
                s_peptide, s_protein,
                T_peptide, T_protein,
                peptide_mask, protein_mask,
            )

            s_peptide = preds["states"]
            T_peptide = Rigid.from_tensor_7(preds["unscaled_frames"])

            outputs.append(preds)

        outputs = dict_multimap(torch.stack, outputs)

        # unslice the output
        masked_angles = torch.tensor([[1.0, 0.0] for _ in range(7)], device=s_peptide.device)
        result = {}
        result["single"] = self._restore_masked(outputs["states"][-1], peptide_slice)
        result["final_frames"] = self._restore_masked(outputs["frames"][-1], peptide_slice)
        result["final_sidechain_frames"] = self._restore_masked(outputs["sidechain_frames"][-1], peptide_slice)
        result["final_angles"] = self._restore_masked(outputs["angles"][-1], peptide_slice, masked_angles)
        result["final_unnormalized_angles"] = self._restore_masked(outputs["unnormalized_angles"][-1], peptide_slice, masked_angles)
        result["final_positions"] = self._restore_masked(outputs["positions"][-1], peptide_slice)
        return result

    @staticmethod
    def _restore_masked(
        residue_value: torch.Tensor,
        residue_slice: torch.Tensor,
        masked_value: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            residue_value:          [*, length, ...]
            residue_slice:          [max_length] (bool)
        Returns:
            masked_residue_value:   [*, max_length, ...]
        """

        dimensions = list(residue_value.shape)
        dimensions[1] = residue_slice.shape[0]

        if masked_value is None:
            masked_residue_value = residue_value.new_zeros(dimensions)
        else:
            masked_residue_value = masked_value.unsqueeze(0).unsqueeze(1).expand(dimensions).clone()

        masked_residue_value[:, residue_slice] = residue_value

        return masked_residue_value

    def _block(self,
               s_peptide_initial: torch.Tensor,
               peptide_aatype: torch.Tensor,
               s_peptide: torch.Tensor,
               s_protein: torch.Tensor,
               T_peptide: Rigid,
               T_protein: Rigid,
               peptide_mask: torch.Tensor,
               protein_mask: torch.Tensor) -> Dict[str, torch.Tensor]:

        # [*, peptide_maxlen, c_s]
        s_upd, ipa_att = self.peptide_ipa(
            s_peptide, s_protein,
            T_peptide, T_protein,
            peptide_mask, protein_mask,
        )
        s_peptide = s_peptide + s_upd
        s_peptide = self.peptide_ipa_dropout(s_peptide)
        s_peptide = self.peptide_layer_norm_ipa(s_peptide)
        s_peptide = self.peptide_transition(s_peptide)

        # [*, peptide_maxlen]
        T_peptide = T_peptide.compose_q_update_vec(self.bb_update(s_peptide))

        # openfold: To hew as closely as possible to AlphaFold, we convert our
        # quaternion-based transformations to rotation-matrix ones
        # here
        backb_to_global = Rigid(
            Rotation(
                rot_mats=T_peptide.get_rots().get_rot_mats(), 
                quats=None
            ),
            T_peptide.get_trans(),
        )

        backb_to_global = backb_to_global.scale_translation(self.trans_scale_factor)

        # [*, peptide_len, 7, 2]
        unnormalized_angles, angles = self.angle_resnet(s_peptide, s_peptide_initial)

        # Calculate frames for side chains
        all_frames_to_global = self.torsion_angles_to_frames(
            backb_to_global,
            angles,
            peptide_aatype,
        )

        # Compute all atom coordinates, from torsions
        pred_xyz = self.frames_and_literature_positions_to_atom14_pos(
            all_frames_to_global,
            peptide_aatype,
        )

        # calculate the actual omega angles, according to atom positions
        post_omegas_from_xyz = self.calculate_omegas_from_positions(pred_xyz, peptide_mask)
        last_omega = post_omegas_from_xyz.new_tensor([0.0, -1.0])  # sine 0, cosine -1 : 180 degrees
        last_omega = last_omega.unsqueeze(0).expand(post_omegas_from_xyz.shape[0], -1).unsqueeze(1)
        omegas = torch.cat([post_omegas_from_xyz, last_omega], dim=-2)
        angles = torch.cat([omegas.unsqueeze(-2), angles[..., 1:, :]], dim=-2)
        unnormalized_angles = torch.cat([omegas.unsqueeze(-2), unnormalized_angles[..., 1:, :]], dim=-2)

        scaled_T_peptide = T_peptide.scale_translation(self.trans_scale_factor)

        preds = {
            "cross_ipa_att": ipa_att,
            "unscaled_frames": T_peptide.to_tensor_7(),
            "frames": scaled_T_peptide.to_tensor_7(),
            "sidechain_frames": all_frames_to_global.to_tensor_4x4(),
            "unnormalized_angles": unnormalized_angles,
            "angles": angles,
            "positions": pred_xyz,
            "states": s_peptide,
        }

        T_peptide = T_peptide.stop_rot_gradient()

        return preds

    def _init_residue_constants(self, float_dtype, device):
        if not hasattr(self, "default_frames"):
            self.register_buffer(
                "default_frames",
                torch.tensor(
                    restype_rigid_group_default_frame,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "group_idx"):
            self.register_buffer(
                "group_idx",
                torch.tensor(
                    restype_atom14_to_rigid_group,
                    device=device,
                    dtype=torch.long,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "atom_mask"):
            self.register_buffer(
                "atom_mask",
                torch.tensor(
                    restype_atom14_mask,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "lit_positions"):
            self.register_buffer(
                "lit_positions",
                torch.tensor(
                    restype_atom14_rigid_group_positions,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )

    def torsion_angles_to_frames(self, T, alpha, aatype):

        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(alpha.dtype, alpha.device)

        # Separated purely to make testing less annoying
        return torsion_angles_to_frames(T, alpha, aatype, self.default_frames)

    def frames_and_literature_positions_to_atom14_pos(
        self, T, aatype  # [*, N, 8]  # [*, N]
    ):
        # Lazily initialize the residue constants on the correct device
        T_rots = T.get_rots()
        self._init_residue_constants(T_rots.dtype, T_rots.device)

        return frames_and_literature_positions_to_atom14_pos(
            T,
            aatype,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions,
        )

    def calculate_omegas_from_positions(self, positions: torch.Tensor, res_mask: torch.Tensor):
        """
        The amide's hydrogen is missing.
        So we calculate the omega from the Ca-C-N-Ca angle.

        Args:
            positions: [*, N_res, 14, 3]
            res_mask:  [*, N_res] (boolean)
        Returns:
            post omegas sin, cos:  [*, N_res - 1, 2] (normalized)
        """

        atom_index_N = 0
        atom_index_CA = 1
        atom_index_C = 2

        # find the atoms

        # [*, N_res - 1, 3]
        positions_CA0 = positions[..., :-1, atom_index_CA, :]
        positions_C0 = positions[..., :-1:, atom_index_C, :]
        positions_N1 = positions[..., 1:, atom_index_N, :]
        positions_CA1 = positions[..., 1:, atom_index_CA, :]

        # [*, N_res - 1]
        mask = torch.logical_and(res_mask[..., :-1], res_mask[..., 1:])
        masked_out = torch.logical_not(mask)

        # make directional vectors for the 3 bonds

        # [*, N_res - 1, 3]
        vec_CCA0 = positions_CA0 - positions_C0

        # [*, N_res - 1, 3]
        vec_NCA1 = positions_CA1 - positions_N1

        # [*, N_res - 1, 3]
        vec_CN = positions_N1 - positions_C0

        # make the newmann projections of the C-alphas on the CN bond

        # [*, N_res - 1, 3]
        plane_n = torch.nn.functional.normalize(vec_CN, dim=-1)

        # [*, N_res - 1, 3]
        newmann0 = torch.nn.functional.normalize(vec_CCA0 - (plane_n * vec_CCA0).sum(dim=-1).unsqueeze(-1) * plane_n, dim=-1)

        # [*, N_res - 1, 3]
        newmann1 = torch.nn.functional.normalize(vec_NCA1 - (plane_n * vec_NCA1).sum(dim=-1).unsqueeze(-1) * plane_n, dim=-1)

        # convert the projections to cosine and sine

        # [*, N_res - 1]
        omega_cos = (newmann0 * newmann1).sum(dim=-1)

        # [*, N_res - 1, 3]
        cross01 = torch.linalg.cross(newmann0, newmann1, dim=-1)

        # [*, N_res - 1]
        omega_sin = torch.norm(cross01, dim=-1)
        omega_sin = torch.where((cross01 * plane_n).sum(dim=-1) < 0.0, -omega_sin, omega_sin)

        # masked areas get 180 degrees omega
        omega_cos = torch.where(mask, omega_cos,-1.0)
        omega_sin = torch.where(mask, omega_sin, 0.0)

        return torch.cat([omega_sin[..., None], omega_cos[..., None]], dim=-1)
