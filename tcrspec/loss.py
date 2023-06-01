import os
import sys
import logging
from uuid import uuid4
from typing import Tuple, Union, Optional, List, Dict, Set
import random
from math import log, sqrt

import ml_collections
import pandas
import torch
from torch.nn import CrossEntropyLoss, MSELoss

from Bio.PDB.PDBIO import PDBIO

from openfold.np.residue_constants import (restype_atom14_ambiguous_atoms as openfold_restype_atom14_ambiguous_atoms,
                                           atom_types as openfold_atom_types,
                                           van_der_waals_radius as openfold_van_der_waals_radius,
                                           make_atom14_dists_bounds as openfold_make_atom14_dists_bounds,
                                           atom_order as openfold_atom_order,
                                           restype_num as openfold_restype_num,
                                           chi_pi_periodic as openfold_chi_pi_periodic)
from openfold.utils.rigid_utils import Rigid
from openfold.utils.tensor_utils import masked_mean as openfold_masked_mean
from openfold.utils.feats import atom14_to_atom37 as openfold_atom14_to_atom37
from openfold.utils.loss import (violation_loss as openfold_compute_violation_loss,
                                 within_residue_violations as openfold_within_residue_violations,
                                 lddt_loss as openfold_compute_lddt_loss,
                                 compute_renamed_ground_truth as openfold_compute_renamed_ground_truth,
                                 backbone_loss as openfold_compute_backbone_loss,
                                 sidechain_loss as openfold_compute_sidechain_loss,
                                 supervised_chi_loss as openfold_supervised_chi_loss,
                                 find_structural_violations as openfold_find_structural_violations,
                                 between_residue_clash_loss as openfold_between_residue_clash_loss,
                                 between_residue_bond_loss as openfold_between_residue_bond_loss,
                                 softmax_cross_entropy as openfold_softmax_cross_entropy)
from openfold.data.data_transforms import (atom37_to_frames as openfold_atom37_to_frames,
                                           make_atom14_masks as openfold_make_atom14_masks)
from openfold.utils.rigid_utils import Rigid, Rotation
from openfold.config import config as openfold_config
from openfold.utils.tensor_utils import permute_final_dims

from tcrspec.time import Timer
from tcrspec.preprocess import preprocess
from tcrspec.dataset import ProteinLoopDataset
from tcrspec.modules.predictor import Predictor
from tcrspec.models.complex import ComplexClass
from tcrspec.models.amino_acid import AminoAcid
from tcrspec.tools.amino_acid import one_hot_decode_sequence
from tcrspec.models.data import TensorDict
from tcrspec.tools.pdb import recreate_structure
from tcrspec.domain.amino_acid import amino_acids_by_one_hot_index


_log = logging.getLogger(__name__)

def _compute_fape_loss(output: TensorDict, batch: TensorDict,
                       config: ml_collections.ConfigDict) -> torch.Tensor:
    """
    Compute FAPE loss as in openfold
    """

    # compute backbone loss
    batch_size, loop_len = batch["loop_len_mask"].shape
    batch_size, protein_len = batch["protein_len_mask"].shape

    bb_mask = torch.cat((batch["protein_len_mask"], batch["loop_len_mask"]), dim=1)
    bb_true_frames = torch.cat((batch["protein_backbone_rigid_tensor"], batch["loop_backbone_rigid_tensor"]), dim=1)
    bb_output_frames = torch.cat((Rigid.from_tensor_4x4(batch["protein_backbone_rigid_tensor"]).to_tensor_7(),
                                  output["final_frames"]), dim=1)

    bb_loss = openfold_compute_backbone_loss(backbone_rigid_tensor=bb_true_frames,
                                             backbone_rigid_mask=bb_mask,
                                             traj=bb_output_frames,
                                             **config.backbone)


    # compute sidechain loss
    atom14_atom_is_ambiguous = torch.tensor(openfold_restype_atom14_ambiguous_atoms[batch["loop_aatype"].cpu().numpy()],
                                            device=batch["loop_aatype"].device)

    renamed_truth = openfold_compute_renamed_ground_truth({
                                                            "atom14_gt_positions": batch["loop_atom14_gt_positions"],
                                                            "atom14_alt_gt_positions": batch["loop_atom14_alt_gt_positions"],
                                                            "atom14_gt_exists": batch["loop_atom14_gt_exists"].float(),
                                                            "atom14_atom_is_ambiguous": atom14_atom_is_ambiguous,
                                                            "atom14_alt_gt_exists": batch["loop_atom14_gt_exists"].float(),
                                                          },
                                                          output["final_positions"])

    truth_frames = openfold_atom37_to_frames(
        {
            "aatype": batch["loop_aatype"],
            "all_atom_positions": batch["loop_all_atom_positions"],
            "all_atom_mask": batch["loop_all_atom_mask"],
        }
    )

    sc_loss = openfold_compute_sidechain_loss(sidechain_frames=output["final_sidechain_frames"][None, ...],
                                              sidechain_atom_pos=output["final_positions"][None, ...],
                                              rigidgroups_gt_frames=truth_frames["rigidgroups_gt_frames"],
                                              rigidgroups_alt_gt_frames=truth_frames["rigidgroups_alt_gt_frames"],
                                              rigidgroups_gt_exists=truth_frames["rigidgroups_gt_exists"],

                                              renamed_atom14_gt_positions=renamed_truth["renamed_atom14_gt_positions"],
                                              renamed_atom14_gt_exists=renamed_truth["renamed_atom14_gt_exists"],
                                              alt_naming_is_better=renamed_truth["alt_naming_is_better"],
                                              **config.sidechain)

    loss = 0.5 * bb_loss + 0.5 * sc_loss

    return loss


def _compute_cross_distance_loss(output: TensorDict, batch: TensorDict) -> torch.Tensor:

    # <might change this later>
    # mse is too inflexible
    # might use bin-based distances instead

    # get xyz from dataset: [batch_size, len, 3]
    true_ca_positions_loop = batch["loop_atom14_gt_positions"][..., 1, :]
    true_ca_positions_protein = batch["protein_atom14_gt_positions"][..., 1, :]
    pred_ca_positions_loop = output["final_positions"][..., 1, :]

    # calculate square distances: [batch_size, loop_len, protein_len]
    true_dist2 = torch.sum(torch.square(true_ca_positions_loop[:, :, None, :] - true_ca_positions_protein[:, None, :, :]), dim=3)
    pred_dist2 = torch.sum(torch.square(pred_ca_positions_loop[:, :, None, :] - true_ca_positions_protein[:, None, :, :]), dim=3)

    # calculate error from existing atoms' distances
    mask = batch["loop_atom14_gt_exists"][:, :, None, 1] * batch["protein_atom14_gt_exists"][:, None, :, 1]
    err = (mask * torch.abs(pred_dist2 - true_dist2)).sum(dim=(1, 2)) / torch.sum(mask.float(), dim=(1, 2))

    return err


def _compute_cross_violation_loss(output: TensorDict, batch: TensorDict,
                                  config: ml_collections.ConfigDict) -> torch.Tensor:

    # Compute the between residue clash loss. (include both loop and protein)
    residx_atom14_to_atom37 = torch.cat((batch["loop_residx_atom14_to_atom37"], batch["protein_residx_atom14_to_atom37"]), dim=1)

    atom14_pred_positions = torch.cat((output["final_positions"], batch["protein_atom14_gt_positions"]), dim=1)
    atom14_atom_exists = torch.cat((batch["loop_atom14_gt_exists"], batch["protein_atom14_gt_exists"]), dim=1)

    # Compute the Van der Waals radius for every atom
    # (the first letter of the atom name is the element type).
    # Shape: (N, 14).
    atomtype_radius = [
        openfold_van_der_waals_radius[name[0]]
        for name in openfold_atom_types
    ]
    atomtype_radius = atom14_pred_positions.new_tensor(atomtype_radius)

    atom14_atom_radius = atom14_atom_exists * atomtype_radius[residx_atom14_to_atom37]
    residue_index = torch.cat((batch["loop_residue_index"], batch["protein_residue_index"]), dim=1)

    between_residue_clashes = openfold_between_residue_clash_loss(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=atom14_atom_exists,
        atom14_atom_radius=atom14_atom_radius,
        residue_index=residue_index,
        overlap_tolerance_soft=config.clash_overlap_tolerance,
        overlap_tolerance_hard=config.clash_overlap_tolerance,
    )

    # Compute all within-residue violations: clashes, bond length and angle violations.
    # (only within loop)
    restype_atom14_bounds = openfold_make_atom14_dists_bounds(
        overlap_tolerance=config.clash_overlap_tolerance,
        bond_length_tolerance_factor=config.violation_tolerance_factor,
    )
    atom14_dists_lower_bound = atom14_pred_positions.new_tensor(
        restype_atom14_bounds["lower_bound"]
    )[batch["loop_aatype"]]
    atom14_dists_upper_bound = atom14_pred_positions.new_tensor(
        restype_atom14_bounds["upper_bound"]
    )[batch["loop_aatype"]]

    residue_violations = openfold_within_residue_violations(
        atom14_pred_positions=output["final_positions"],
        atom14_atom_exists=batch["loop_atom14_gt_exists"],
        atom14_dists_lower_bound=atom14_dists_lower_bound,
        atom14_dists_upper_bound=atom14_dists_upper_bound,
        tighten_bounds_for_loss=0.0,
    )

    # Compute between residue backbone violations of bonds and angles.
    connection_violations = openfold_between_residue_bond_loss(
        pred_atom_positions=output["final_positions"],
        pred_atom_mask=batch["loop_atom14_gt_exists"],
        residue_index=batch["loop_residue_index"],
        aatype=batch["loop_aatype"],
        tolerance_factor_soft=config.violation_tolerance_factor,
        tolerance_factor_hard=config.violation_tolerance_factor,
    )

    # []
    violations_between_residues_bonds_c_n_loss_mean = connection_violations["c_n_loss_mean"]
    violations_between_residues_angles_ca_c_n_loss_mean = connection_violations["ca_c_n_loss_mean"]
    violations_between_residues_angles_c_n_ca_loss_mean = connection_violations["c_n_ca_loss_mean"]

    # [loop_len + protein_len, 14]
    violations_between_residues_clashes_per_atom_loss_sum = between_residue_clashes["per_atom_loss_sum"]

    # [loop_len, 14]
    violations_within_residues_per_atom_loss_sum = residue_violations["per_atom_loss_sum"]

    # Calculate loss, as in openfold
    loop_num_atoms = torch.sum(batch["loop_atom14_gt_exists"])
    num_atoms = loop_num_atoms + torch.sum(batch["protein_atom14_gt_exists"])

    l_clash = torch.sum(violations_between_residues_clashes_per_atom_loss_sum) / (config.eps + num_atoms) + \
              torch.sum(violations_within_residues_per_atom_loss_sum) / (config.eps + loop_num_atoms)

    loss = (
        violations_between_residues_bonds_c_n_loss_mean +
        violations_between_residues_angles_ca_c_n_loss_mean +
        violations_between_residues_angles_c_n_ca_loss_mean +
        l_clash
    )

    return loss


def _compute_cross_lddt_loss(output: TensorDict, batch: TensorDict,
                             config: ml_collections.ConfigDict) -> torch.Tensor:

    ca_index = openfold_atom_order["CA"]

    protein_positions = batch["protein_all_atom_positions"][..., ca_index, :]
    protein_positions_mask = batch["protein_all_atom_mask"][..., ca_index : (ca_index + 1)]
    loop_positions_mask = batch["loop_all_atom_mask"][..., ca_index : (ca_index + 1)]
    true_loop_positions = batch["loop_all_atom_positions"][..., ca_index, :]
    pred_loop_positions = output["final_atom_positions"][..., ca_index, :]

    logits = output["lddt_logits"]

    # compute lddt from loop to protein
    dmat_true = torch.sqrt(
        config.eps + torch.sum((true_loop_positions[..., None, :] - protein_positions[..., None, :, :]) ** 2, dim=-1)
    )

    dmat_pred = torch.sqrt(
        config.eps + torch.sum((pred_loop_positions[..., None, :] - protein_positions[..., None, :, :]) ** 2, dim=-1)
    )

    dists_to_score = (
        (dmat_true < config.cutoff)
        * loop_positions_mask * protein_positions_mask.transpose(-1, -2)
    )

    dist_l1 = torch.abs(dmat_true - dmat_pred)

    score = (
        (dist_l1 < 0.5).type(dist_l1.dtype)
        + (dist_l1 < 1.0).type(dist_l1.dtype)
        + (dist_l1 < 2.0).type(dist_l1.dtype)
        + (dist_l1 < 4.0).type(dist_l1.dtype)
    ) * 0.25

    norm = 1.0 / (config.eps + torch.sum(dists_to_score, dim=-1))
    score = norm * (config.eps + torch.sum(dists_to_score * score, dim=-1))

    # compute lddt loss
    score = score.detach()

    bin_index = torch.floor(score * config.no_bins).long()
    bin_index = torch.clamp(bin_index, max=(config.no_bins - 1))
    lddt_ca_one_hot = torch.nn.functional.one_hot(bin_index, num_classes=config.no_bins)

    errors = openfold_softmax_cross_entropy(logits, lddt_ca_one_hot)
    all_atom_mask = loop_positions_mask.squeeze(-1)
    loss = torch.sum(errors * all_atom_mask, dim=-1) / (config.eps + torch.sum(all_atom_mask, dim=-1))

    # Average over the batch dimension
    return loss.mean()


def _supervised_chi_loss(angles_sin_cos: torch.Tensor,
                         unnormalized_angles_sin_cos: torch.Tensor,
                         aatype: torch.Tensor,
                         seq_mask: torch.Tensor,
                         chi_mask: torch.Tensor,
                         chi_angles_sin_cos: torch.Tensor,
                         chi_weight: float,
                         angle_norm_weight: float,
                         eps=1e-6,
                         **kwargs,
) -> torch.Tensor:
    """
        Implements Algorithm 27 (torsionAngleLoss)

        Args:
            angles_sin_cos:
                [*, N, 7, 2] predicted angles
            unnormalized_angles_sin_cos:
                The same angles, but unnormalized
            aatype:
                [*, N] residue indices
            seq_mask:
                [*, N] sequence mask
            chi_mask:
                [*, N, 7] angle mask
            chi_angles_sin_cos:
                [*, N, 7, 2] ground truth angles
            chi_weight:
                Weight for the angle component of the loss
            angle_norm_weight:
                Weight for the normalization component of the loss
        Returns:
            [*] loss tensor
    """
    pred_angles = angles_sin_cos[..., 3:, :]
    residue_type_one_hot = torch.nn.functional.one_hot(
        aatype,
        openfold_restype_num + 1,
    )
    chi_pi_periodic = torch.einsum(
        "...ij,jk->ik",
        residue_type_one_hot.type(angles_sin_cos.dtype),
        angles_sin_cos.new_tensor(openfold_chi_pi_periodic),
    )

    true_chi = chi_angles_sin_cos[None]

    shifted_mask = (1 - 2 * chi_pi_periodic).unsqueeze(-1)
    true_chi_shifted = shifted_mask * true_chi
    sq_chi_error = torch.sum((true_chi - pred_angles) ** 2, dim=-1)
    sq_chi_error_shifted = torch.sum(
        (true_chi_shifted - pred_angles) ** 2, dim=-1
    )
    sq_chi_error = torch.minimum(sq_chi_error, sq_chi_error_shifted)

    # The ol' switcheroo
    sq_chi_error = sq_chi_error.permute(
        *range(len(sq_chi_error.shape))[1:-2], 0, -2, -1
    )

    sq_chi_loss = openfold_masked_mean(
        chi_mask[..., None, :, :], sq_chi_error, dim=(-1, -2, -3)
    )

    loss = chi_weight * sq_chi_loss

    angle_norm = torch.sqrt(
        torch.sum(unnormalized_angles_sin_cos ** 2, dim=-1) + eps
    )
    norm_error = torch.abs(angle_norm - 1.0)
    norm_error = norm_error.permute(
        *range(len(norm_error.shape))[1:-2], 0, -2, -1
    )
    angle_norm_loss = openfold_masked_mean(
        seq_mask[..., :, None], norm_error, dim=(-1, -2, -3)
    )

    loss = loss + angle_norm_weight * angle_norm_loss

    return loss


_affinity_binding_treshold = 1.0 - log(500) / log(50000)
_affinity_loss_function = MSELoss(reduction="none")


def get_loss(output: TensorDict, batch: TensorDict,
             fine_tune: bool) -> TensorDict:

    # compute our own affinity-based loss
    affinity_loss = _affinity_loss_function(output["affinity"], batch["affinity"])

    # compute chi loss, as in openfold
    chi_loss = _supervised_chi_loss(output["final_angles"],
                                    output["final_unnormalized_angles"],
                                    batch["loop_aatype"],
                                    batch["loop_len_mask"],
                                    batch["loop_torsion_angles_mask"][..., 3:],
                                    batch["loop_torsion_angles_sin_cos"][..., 3:, :],
                                    **openfold_config.loss.supervised_chi)

    # compute fape loss, as in openfold
    fape_loss = _compute_fape_loss(output, batch,
                                   openfold_config.loss.fape)

    # compute violations loss, using an adjusted function
    violation_loss = _compute_cross_violation_loss(output, batch, openfold_config.loss.violation)

    # combine the loss terms
    total_loss = 1.0 * affinity_loss + \
                 1.0 * chi_loss + \
                 1.0 * fape_loss

    if fine_tune:
        total_loss += 1.0 * violation_loss

    # for true non-binders, the total loss is simply affinity-based
    non_binders_index = batch["affinity"] < _affinity_binding_treshold
    total_loss[non_binders_index] = 1.0 * affinity_loss[non_binders_index]

    # average losses over batch dimension
    return TensorDict({
        "total": total_loss.mean(dim=0),
        "affinity": affinity_loss.mean(dim=0),
        "fape": fape_loss.mean(dim=0),
        "chi": chi_loss.mean(dim=0),
        "violation": violation_loss.mean(dim=0),
    })
