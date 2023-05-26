from typing import Dict, Union
import logging

import torch

from openfold.utils.rigid_utils import Rotation, Rigid
from openfold.model.primitives import Linear, LayerNorm
from openfold.model.structure_module import BackboneUpdate, AngleResnet, StructureModuleTransition
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

from .cross_ipa import CrossInvariantPointAttention


_log = logging.getLogger(__name__)


class CrossStructureModule(torch.nn.Module):
    def __init__(
        self,
        c_s,
        c_z,
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

        self.layer_norm_s_loop = LayerNorm(self.c_s)
        self.layer_norm_s_protein = LayerNorm(self.c_s)

        self.linear_in_loop = Linear(self.c_s, self.c_s)
        self.linear_in_protein = Linear(self.c_s, self.c_s)

        self.ipa = CrossInvariantPointAttention(
            self.c_s,
            self.c_ipa,
            self.no_heads_ipa,
            self.no_qk_points,
            self.no_v_points,
            eps=self.epsilon,
        )

        self.ipa_dropout = torch.nn.Dropout(self.dropout_rate)
        self.layer_norm_ipa = LayerNorm(self.c_s)
        self.transition = StructureModuleTransition(self.c_s,
                                                    self.n_transition_layers,
                                                    self.dropout_rate)

        self.bb_update = BackboneUpdate(self.c_s)

        self.angle_resnet = AngleResnet(
            self.c_s,
            self.c_resnet,
            self.no_resnet_blocks,
            self.no_angles,
            self.epsilon,
        )

    def forward(
        self,
        loop_aatype: torch.Tensor,
        s_loop_initial: torch.Tensor,
        loop_mask: torch.Tensor,
        s_protein_initial: torch.Tensor,
        protein_mask: torch.Tensor,
        T_protein: Rigid

    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            loop_aatype:       [batch_size, loop_len]
            s_loop_initial:    [batch_size, loop_len, c_s]
            loop_mask:         [batch_size, loop_len]
            s_protein_initial: [batch_size, protein_len, c_s]
            protein_mask:      [batch_size, protein_len]
            T_protein:         [batch_size, protein_len, 4, 4]
        Returns:
            frames:              [n_blocks, batch_size, loop_len, 4, 4]
            sidechain_frames:    [n_blocks, batch_size, loop_len, 4, 4]
            unnormalized_angles: [n_blocks, batch_size, loop_len, 7, 2]
            angles:              [n_blocks, batch_size, loop_len, 7, 2]
            positions:           [n_blocks, batch_size, loop_len, 18, 3]
            states:              [n_blocks, batch_size, loop_len, c_s]
            single:           [batch_size, loop_len, c_s]
        """

        s_loop = torch.clone(s_loop_initial)
        s_protein = torch.clone(s_protein_initial)

        # [batch_size, loop_len, c_s]
        s_loop = self.layer_norm_s_loop(s_loop)

        # [batch_size, protein_len, c_s]
        s_protein = self.layer_norm_s_protein(s_protein)

        # [batch_size, loop_len, c_s]
        s_loop_initial = torch.clone(s_loop)
        s_loop = self.linear_in_loop(s_loop)

        # [batch_size, protein_len, c_s]
        s_protein_initial = torch.clone(s_protein)
        s_protein = self.linear_in_protein(s_protein)

        # [batch_size, loop_len]
        T_loop = Rigid.identity(
            s_loop.shape[:-1], 
            s_loop.dtype, 
            s_loop.device, 
            self.training,
            fmt="quat",
        )

        outputs = []
        for i in range(self.n_blocks):

            s_loop, T_loop, preds, att = self._block(s_loop_initial,
                                                     loop_aatype,
                                                     s_loop, s_protein,
                                                     T_loop, T_protein,
                                                     loop_mask, protein_mask)

            outputs.append(preds)

        outputs = dict_multimap(torch.stack, outputs)

        r = {}
        r["single"] = s_loop
        r["cross_attention"] = att
        r["final_frames"] = outputs["frames"][-1]
        r["final_sidechain_frames"] = outputs["sidechain_frames"][-1]
        r["final_angles"] = outputs["angles"][-1]
        r["final_unnormalized_angles"] = outputs["unnormalized_angles"][-1]
        r["final_positions"] = outputs["positions"][-1]

        return r

    def _block(self,
               s_loop_initial: torch.Tensor,
               loop_aatype: torch.Tensor,
               s_loop: torch.Tensor,
               s_protein: torch.Tensor,
               T_loop: Rigid,
               T_protein: Rigid,
               loop_mask: torch.Tensor,
               protein_mask: torch.Tensor) -> Dict[str, torch.Tensor]:

        # [batch_size, loop_len, c_s]
        s_upd, ipa_att = self.ipa(
            s_loop, s_protein,
            T_loop, T_protein,
            loop_mask, protein_mask
        )
        s_loop = s_loop + s_upd
        s_loop = self.ipa_dropout(s_loop)
        s_loop = self.layer_norm_ipa(s_loop)
        s_loop = self.transition(s_loop)

        # [batch_size, loop_len]
        T_loop = T_loop.compose_q_update_vec(self.bb_update(s_loop))

        # To hew as closely as possible to AlphaFold, we convert our
        # quaternion-based transformations to rotation-matrix ones
        # here
        backb_to_global = Rigid(
            Rotation(
                rot_mats=T_loop.get_rots().get_rot_mats(), 
                quats=None
            ),
            T_loop.get_trans(),
        )

        backb_to_global = backb_to_global.scale_translation(
            self.trans_scale_factor
        )

        # [batch_size, loop_len, 7, 2]
        unnormalized_angles, angles = self.angle_resnet(s_loop, s_loop_initial)

        # Calculate frames for side chains
        all_frames_to_global = self.torsion_angles_to_frames(
            backb_to_global,
            angles,
            loop_aatype,
        )

        # Compute all atom coordinates, from torsions
        pred_xyz = self.frames_and_literature_positions_to_atom14_pos(
            all_frames_to_global,
            loop_aatype,
        )

        scaled_T_loop = T_loop.scale_translation(self.trans_scale_factor)

        preds = {
            "frames": scaled_T_loop.to_tensor_7(),
            "sidechain_frames": all_frames_to_global.to_tensor_4x4(),
            "unnormalized_angles": unnormalized_angles,
            "angles": angles,
            "positions": pred_xyz,
            "states": s_loop,
        }

        T_loop = T_loop.stop_rot_gradient()

        return s_loop, T_loop, preds, ipa_att

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
