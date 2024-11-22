from typing import Dict, Union, Optional
import logging

import torch

import ml_collections

from openfold.utils.rigid_utils import Rotation
from openfold.model.primitives import Linear, LayerNorm
from openfold.model.structure_module import AngleResnet, StructureModuleTransition, BackboneUpdate
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


_log = logging.getLogger(__name__)


class CrossStructureModule(torch.nn.Module):
    """
    This is like algorithm 20 in AlphaFold2, but with some modifications:

     - omega angles are calculated from predicted frames. They are not predicted directly.
     - It does not predict frames for a complete sequence. It takes a protein structure and peptide sequence as input. The peptide structure is predicted.

    The code was copied from OpenFold and then modified.
    The original StructureModule class is at:
    https://github.com/aqlaboratory/openfold/blob/main/openfold/model/structure_module.py
    """

    def __init__(self, config: ml_collections.ConfigDict):
        """
        in config:
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
        self.c_s = config.c_s
        self.c_ipa = config.c_hidden
        self.c_resnet = config.c_resnet
        self.no_heads_ipa = config.no_heads
        self.no_qk_points = config.no_qk_points
        self.no_v_points = config.no_v_points
        self.dropout_rate = config.dropout_rate
        self.n_blocks = config.no_cross_blocks
        self.no_resnet_blocks = config.no_resnet_blocks
        self.no_angles = config.no_angles
        self.trans_scale_factor = config.trans_scale_factor
        self.epsilon = config.epsilon
        self.n_transition_layers = config.no_transition_layers

        # Buffers to be lazily initialized later in _init_residue_constants once the dtype and device are determined.
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
        self.cross_ipa = CrossInvariantPointAttention(config)
        self.cross_ipa_dropout = torch.nn.Dropout(self.dropout_rate)
        self.cross_ipa_norm = LayerNorm(self.c_s)
        self.cross_ipa_transition = StructureModuleTransition(self.c_s,
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

        # init the angle bias to be nonzero
        self._init_angle()

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
        This method predicts peptide structure.

        Args:
            peptide_aatype:         [*, peptide_maxlen] (int, 0 - 19, sequence)
            s_peptide_initial:      [*, peptide_maxlen, c_s] (sequence embedding)
            peptide_mask:           [*, peptide_maxlen] (whether to use the residue or not)
            s_protein_initial:      [*, protein_maxlen, c_s] (sequence embedding)
            protein_mask:           [*, protein_maxlen] (whether to use the residue or not)
            T_protein:              [*, protein_maxlen, 4, 4] (backbone frames)
        Returns:
            frames:                 [*, peptide_maxlen, 7] (backbone frames)
            sidechain_frames:       [*, peptide_maxlen, 7, 4, 4] (sidechain frames)
            unnormalized_angles:    [*, peptide_maxlen, 7, 2] (sin,cos but not normalized)
            angles:                 [*, peptide_maxlen, 7, 2] (sin,cos and normalized)
            positions:              [*, peptide_maxlen, 14, 3] (atom positions in 14-atom format)
            single:                 [*, peptide_maxlen, c_s] (sequence embedding)
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

            # retrieve updated state of s_peptide and T_peptide from block_results
            s_peptide = preds["states"]
            T_peptide = Rigid.from_tensor_7(preds["unscaled_frames"])

            outputs.append(preds)

        outputs = dict_multimap(torch.stack, outputs)

        # fill in an angle vector of length 1.0 for masked out parts, to prevent NaN in the loss function's normalization
        masked_angles = torch.tensor([[1.0, 0.0] for _ in range(7)], device=s_peptide.device)
        # unslice the output to a returned dictionary
        result = {}
        result["single"] = self._unslice_and_restore_masked(outputs["states"][-1], peptide_slice)
        result["final_frames"] = self._unslice_and_restore_masked(outputs["frames"][-1], peptide_slice)
        result["final_sidechain_frames"] = self._unslice_and_restore_masked(outputs["sidechain_frames"][-1], peptide_slice)
        result["final_angles"] = self._unslice_and_restore_masked(outputs["angles"][-1], peptide_slice, masked_angles)
        result["final_unnormalized_angles"] = self._unslice_and_restore_masked(outputs["unnormalized_angles"][-1], peptide_slice, masked_angles)
        result["final_positions"] = self._unslice_and_restore_masked(outputs["positions"][-1], peptide_slice)
        return result

    @staticmethod
    def _unslice_and_restore_masked(
        residue_value: torch.Tensor,
        residue_slice: torch.Tensor,
        masked_value: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Undoes the effect of slicing.
        Expands the data tensor back into its original size, with masks.

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

    def _block(
        self,
        s_peptide_initial: torch.Tensor,
        peptide_aatype: torch.Tensor,
        s_peptide: torch.Tensor,
        s_protein: torch.Tensor,
        T_peptide: Rigid,
        T_protein: Rigid,
        peptide_mask: torch.Tensor,
        protein_mask: torch.Tensor,

    ) -> Dict[str, torch.Tensor]:
        """
        One iterated block, similar to AlphaFold2 Algorithmn 20, lines 5-31
        """

        # [*, peptide_maxlen, c_s]
        s_upd, ipa_att = self.cross_ipa(
            s_peptide, s_protein,
            T_peptide, T_protein,
            peptide_mask, protein_mask,
        )
        s_peptide = s_peptide + s_upd
        s_peptide = self.cross_ipa_dropout(s_peptide)
        s_peptide = self.cross_ipa_norm(s_peptide)
        s_peptide = self.cross_ipa_transition(s_peptide)

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

        # apply the scale factor, to get the final backbone frames
        backb_to_global = backb_to_global.scale_translation(self.trans_scale_factor)

        # [*, peptide_len, 7, 2]
        unnormalized_angles, angles = self.angle_resnet(s_peptide, s_peptide_initial)

        # Calculate frames for side chains torsion angles
        all_frames_to_global = self.torsion_angles_to_frames(
            backb_to_global,
            angles,
            peptide_aatype,
        )

        # Compute all atom coordinates, from torsion frames
        pred_xyz = self.frames_and_literature_positions_to_atom14_pos(
            all_frames_to_global,
            peptide_aatype,
        )

        # calculate the actual omega angles, according to backbone atom positions
        post_omegas_from_xyz = self._calculate_omegas_from_positions(pred_xyz, peptide_mask)
        last_omega = post_omegas_from_xyz.new_tensor([0.0, -1.0])  # sine 0, cosine -1 : 180 degrees
        last_omega = last_omega.unsqueeze(0).expand(post_omegas_from_xyz.shape[0], -1).unsqueeze(1)
        omegas = torch.cat([post_omegas_from_xyz, last_omega], dim=-2)
        # insert the omegas in the angle tensors
        angles = torch.cat([omegas.unsqueeze(-2), angles[..., 1:, :]], dim=-2)
        unnormalized_angles = torch.cat([omegas.unsqueeze(-2), unnormalized_angles[..., 1:, :]], dim=-2)

        # apply the scale factor, to get the final backbone frames in the output
        scaled_T_peptide = T_peptide.scale_translation(self.trans_scale_factor)

        # unscaled frames are added to this output, to be used in the next block
        preds = {
            "unscaled_frames": T_peptide.to_tensor_7(),  # openfold stores these as vec+quaternion
            "frames": scaled_T_peptide.to_tensor_7(),  # openfold stores these as vec+quaternion
            "sidechain_frames": all_frames_to_global.to_tensor_4x4(),  # openfold stores these 4x4
            "unnormalized_angles": unnormalized_angles,
            "angles": angles,
            "positions": pred_xyz,
            "states": s_peptide,
        }

        T_peptide = T_peptide.stop_rot_gradient()

        return preds

    def _init_angle(self):
        """
        Sometimes, the angle resnet produces (0,0) sin,cos angles.
        This results in a zero norm and NaN loss values.

        To prevent that, we set the bias to nonzero values.
        """

        with torch.no_grad():
            self.angle_resnet.linear_out.bias.fill_(self.epsilon)

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

    def _calculate_omegas_from_positions(self, positions: torch.Tensor, res_mask: torch.Tensor):
        """
        The amide's hydrogen is absent.
        So we calculate the omega from the Ca-C-N-Ca angle.

        Args:
            positions: [*, N_res, 14, 3]
            res_mask:  [*, N_res] (boolean)
        Returns:
            post omegas sin, cos:  [*, N_res - 1, 2] (normalized)
        """

        # positions in the array where the bakcbone atoms are stored:
        atom_index_N = 0
        atom_index_CA = 1
        atom_index_C = 2

        # find the backbone atoms

        # [*, N_res - 1, 3]
        positions_CA0 = positions[..., :-1, atom_index_CA, :]
        positions_C0 = positions[..., :-1:, atom_index_C, :]
        positions_N1 = positions[..., 1:, atom_index_N, :]
        positions_CA1 = positions[..., 1:, atom_index_CA, :]

        # [*, N_res - 1]
        mask = torch.logical_and(res_mask[..., :-1], res_mask[..., 1:])
        masked_out = torch.logical_not(mask)

        # make directional vectors for the 3 bonds: C-alpha---C---N---C-alpha

        # [*, N_res - 1, 3]
        vec_CCA0 = positions_CA0 - positions_C0

        # [*, N_res - 1, 3]
        vec_NCA1 = positions_CA1 - positions_N1

        # [*, N_res - 1, 3]
        vec_CN = positions_N1 - positions_C0

        # make the newmann projections of the C-alphas on the C---N bond

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
