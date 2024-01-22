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
                                           restype_name_to_atom14_names as openfold_restype_name_to_atom14_names,
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
                                 compute_fape as openfold_compute_fape,
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
from tcrspec.models.amino_acid import AminoAcid
from tcrspec.tools.amino_acid import one_hot_decode_sequence
from tcrspec.models.data import TensorDict
from tcrspec.tools.pdb import recreate_structure
from tcrspec.domain.amino_acid import amino_acids_by_one_hot_index


_log = logging.getLogger(__name__)

def _compute_fape_loss(
    output: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    config: ml_collections.ConfigDict
) -> Dict[str, torch.Tensor]:
    """
    Compute FAPE loss as in openfold

    Returns:
        backbone:   [*] backbone FAPE
        sidechain:  [*] sidechain FAPE
        total:      [*] backbone FAPE + sidechain FAPE
    """

    # compute backbone loss
    peptide_mask = batch["peptide_cross_residues_mask"]
    peptide_true_frames = Rigid.from_tensor_4x4(batch["peptide_backbone_rigid_tensor"])
    peptide_output_frames = Rigid.from_tensor_7(output["final_frames"])
    protein_frames = Rigid.from_tensor_4x4(batch["protein_backbone_rigid_tensor"])
    protein_mask = batch["protein_cross_residues_mask"]

    bb_loss = torch.mean(
        openfold_compute_fape(
            pred_frames=protein_frames,
            target_frames=protein_frames,
            frames_mask=protein_mask,
            pred_positions=peptide_output_frames.get_trans(),
            target_positions=peptide_true_frames.get_trans(),
            positions_mask=peptide_mask,
            length_scale=10.0,
            l1_clamp_distance=10.0,
            eps=1e-4,
        )
    )

    # compute sidechain loss
    atom14_atom_is_ambiguous = torch.tensor(openfold_restype_atom14_ambiguous_atoms[batch["peptide_aatype"].cpu().numpy()],
                                            device=batch["peptide_aatype"].device)

    renamed_truth = openfold_compute_renamed_ground_truth({
                                                            "atom14_gt_positions": batch["peptide_atom14_gt_positions"],
                                                            "atom14_alt_gt_positions": batch["peptide_atom14_alt_gt_positions"],
                                                            "atom14_gt_exists": batch["peptide_atom14_gt_exists"].float(),
                                                            "atom14_atom_is_ambiguous": atom14_atom_is_ambiguous,
                                                            "atom14_alt_gt_exists": batch["peptide_atom14_gt_exists"].float(),
                                                          },
                                                          output["final_positions"])

    truth_frames = openfold_atom37_to_frames(
        {
            "aatype": batch["peptide_aatype"],
            "all_atom_positions": batch["peptide_all_atom_positions"],
            "all_atom_mask": batch["peptide_all_atom_mask"],
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

    total_loss = 0.5 * bb_loss + 0.5 * sc_loss

    return {
        "total": total_loss,
        "backbone": bb_loss,
        "sidechain": sc_loss,
    }


def _compute_cross_distance_loss(output: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Calculates the error on the distances between the protein and peptide CAs

    Returns:
        [*] the MSE on the distances
    """

    # get xyz from dataset: [*, len, 3]
    true_ca_positions_peptide = batch["peptide_atom14_gt_positions"][..., 1, :]
    true_ca_positions_protein = batch["protein_atom14_gt_positions"][..., 1, :]
    pred_ca_positions_peptide = output["final_positions"][..., 1, :]

    # calculate square distances: [*, peptide_maxlen, protein_maxlen]
    true_dist2 = torch.sum(torch.square(true_ca_positions_peptide[:, :, None, :] - true_ca_positions_protein[:, None, :, :]), dim=3)
    pred_dist2 = torch.sum(torch.square(pred_ca_positions_peptide[:, :, None, :] - true_ca_positions_protein[:, None, :, :]), dim=3)

    # calculate error from existing atoms' distance
    # [*]
    mask = batch["peptide_atom14_gt_exists"][:, :, None, 1] * batch["protein_atom14_gt_exists"][:, None, :, 1]
    err = (mask * torch.abs(pred_dist2 - true_dist2)).sum(dim=(1, 2)) / torch.sum(mask.float(), dim=(1, 2))

    return err


def _compute_cross_violation_loss(output: Dict[str, torch.Tensor],
                                  batch: Dict[str, torch.Tensor],
                                  config: ml_collections.ConfigDict) -> Dict[str, torch.Tensor]:
    """
    Compute violations in the predicted structure.
    Returns:
        bond:                       [*] bond violations between residues within the peptide
        CA-C-N-angles:              [*] C-alpha-C-N angle violations in peptide
        C-N-CA-angles:              [*] C-N-C-alpha angle violations in peptide
        between-residues-clash:     [*] clashes between residues from protein and peptide
        within-residues-clash:      [*] clashes between atoms within peptide residues
    """

    # Compute the between residue clash loss. (include both peptide and protein)
    # [*, peptide_maxlen + protein_maxlen, 14]
    residx_atom14_to_atom37 = torch.cat((batch["peptide_residx_atom14_to_atom37"],
                                         batch["protein_residx_atom14_to_atom37"]), dim=1)

    # [*, peptide_maxlen + protein_maxlen, 14, 3]
    atom14_pred_positions = torch.cat((output["final_positions"],
                                       batch["protein_atom14_gt_positions"]), dim=1)

    # [*, peptide_maxlen + protein_maxlen, 14]
    atom14_atom_exists = torch.cat((batch["peptide_atom14_gt_exists"],
                                    batch["protein_atom14_gt_exists"]), dim=1)

    # Compute the Van der Waals radius for every atom
    # (the first letter of the atom name is the element type).
    # [37]
    atomtype_radius = [
        openfold_van_der_waals_radius[name[0]]
        for name in openfold_atom_types
    ]
    # [37]
    atomtype_radius = atom14_pred_positions.new_tensor(atomtype_radius)

    # [*, peptide_maxlen + protein_maxlen, 14]
    atom14_atom_radius = atom14_atom_exists * atomtype_radius[residx_atom14_to_atom37]

    peptide_residue_index = batch["peptide_residue_index"]
    protein_residue_index = batch["protein_residue_index"]

    # [*, peptide_maxlen + protein_maxlen]
    residue_index = torch.cat((peptide_residue_index, protein_residue_index), dim=1)

    between_residue_clashes = openfold_between_residue_clash_loss(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=atom14_atom_exists,
        atom14_atom_radius=atom14_atom_radius,
        residue_index=residue_index,
        overlap_tolerance_soft=config.clash_overlap_tolerance,
        overlap_tolerance_hard=config.clash_overlap_tolerance,
    )

    # Compute all within-residue violations: clashes, bond length and angle violations.
    # (only within peptide)
    restype_atom14_bounds = openfold_make_atom14_dists_bounds(
        overlap_tolerance=config.clash_overlap_tolerance,
        bond_length_tolerance_factor=config.violation_tolerance_factor,
    )
    atom14_dists_lower_bound = atom14_pred_positions.new_tensor(
        restype_atom14_bounds["lower_bound"]
    )[batch["peptide_aatype"]]
    atom14_dists_upper_bound = atom14_pred_positions.new_tensor(
        restype_atom14_bounds["upper_bound"]
    )[batch["peptide_aatype"]]

    residue_violations = openfold_within_residue_violations(
        atom14_pred_positions=output["final_positions"],
        atom14_atom_exists=batch["peptide_atom14_gt_exists"],
        atom14_dists_lower_bound=atom14_dists_lower_bound,
        atom14_dists_upper_bound=atom14_dists_upper_bound,
        tighten_bounds_for_loss=0.0,
    )

    # Compute between residue backbone violations of bonds and angles.
    connection_violations = openfold_between_residue_bond_loss(
        pred_atom_positions=output["final_positions"],
        pred_atom_mask=batch["peptide_atom14_gt_exists"],
        residue_index=peptide_residue_index,
        aatype=batch["peptide_aatype"],
        tolerance_factor_soft=config.violation_tolerance_factor,
        tolerance_factor_hard=config.violation_tolerance_factor,
    )

    # []
    violations_between_residues_bonds_c_n_loss_mean = connection_violations["c_n_loss_mean"]
    violations_between_residues_angles_ca_c_n_loss_mean = connection_violations["ca_c_n_loss_mean"]
    violations_between_residues_angles_c_n_ca_loss_mean = connection_violations["c_n_ca_loss_mean"]

    # [peptide_len + protein_len, 14]
    violations_between_residues_clashes_per_atom_loss_sum = between_residue_clashes["per_atom_loss_sum"]

    # [peptide_len, 14]
    violations_within_residues_per_atom_loss_sum = residue_violations["per_atom_loss_sum"]

    # Calculate loss, as in openfold
    peptide_num_atoms = torch.sum(batch["peptide_atom14_gt_exists"])

    between_residues_clash = torch.sum(violations_between_residues_clashes_per_atom_loss_sum) / (config.eps + peptide_num_atoms)
    within_residues_clash = torch.sum(violations_within_residues_per_atom_loss_sum) / (config.eps + peptide_num_atoms)

    loss = {
        "bond": violations_between_residues_bonds_c_n_loss_mean,
        "CA-C-N-angles": violations_between_residues_angles_ca_c_n_loss_mean,
        "C-N-CA-angles": violations_between_residues_angles_c_n_ca_loss_mean,
        "between-residues-clash": between_residues_clash,
        "within-residues-clash": within_residues_clash,
        "total": (violations_between_residues_bonds_c_n_loss_mean +
                  violations_between_residues_angles_ca_c_n_loss_mean +
                  violations_between_residues_angles_c_n_ca_loss_mean +
                  between_residues_clash +
                  within_residues_clash)
    }

    return loss


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
        Implements Algorithm 27, (torsionAngleLoss)
        but unlike the version in openfold. we do not take the average over the batch.
        Instead, this function returns a tensor that returns the loss per peptide.

        Args:
            angles_sin_cos:
                [*, N, 7, 2] predicted angles
            unnormalized_angles_sin_cos:
                [*, N, 7, 2] The same angles, but unnormalized
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


AFFINITY_BINDING_TRESHOLD = 1.0 - log(500) / log(50000)

_classification_loss_function = torch.nn.CrossEntropyLoss(reduction="none")
_regression_loss_function = torch.nn.MSELoss(reduction="none")


def get_loss(output: Dict[str, torch.Tensor],
             batch: Dict[str, torch.Tensor],
             affinity_tune: bool,
             fape_tune: bool,
             chi_tune: bool,
             fine_tune: bool) -> TensorDict:
    """
    Compute all losses and sum them up, according to what is desired.

    Args:
        output: what came out of the model
        batch: what came out of the dataset
        affinity_tune: whether or not to include the binding affinity loss term
        fape_tune: whether or not to include the FAPE loss term
        chi_tune: whether or not to include the chi loss term
        fine_tune: whether or not to include bond length, bond angle violations or clashes in the loss

    Returns:
        total:                      [*] the sum of selected loss terms, per batch entry
        chi:                        [*] chi loss, per batch entry
        affinity:                   [*] binding affinity loss, per batch entry
        total violation:            [*] bond length, bond angle violations and clashes, per batch entry
        bond violation:             [*] mean bond length violation loss, per batch entry
        CA-C-N-angles:              [*] mean C-alpha-C-N angle violations in peptide, per batch entry
        C-N-CA-angles:              [*] mean C-N-C-alpha angle violations in peptide, per batch entry
        between-residues-clash:     [*] mean clashes between residues from protein and peptide, per batch entry
        within-residues-clash:      [*] mean clashes between atoms within peptide residues, per batch entry
        total fape:                 [*] frame aligned point error, per batch entry
        backbone fape:              [*] backbone frame aligned point error, per batch entry
        sidechain fape:             [*] sidechain frame aligned point error, per batch entry
    """

    # compute our own affinity-based loss
    if "affinity" in output and "affinity" in batch:
        affinity_loss = _regression_loss_function(output["affinity"], batch["affinity"])
        non_binders_index = batch["affinity"] < AFFINITY_BINDING_TRESHOLD

    elif "class" in output and "class" in batch:
        affinity_loss = _classification_loss_function(output["logits"], batch["class"])
        non_binders_index = torch.logical_not(batch["class"])

    else:
        raise ValueError("Cannot compute affinity loss without class or affinity in both output and batch data")

    # compute chi loss, as in openfold
    chi_loss = _supervised_chi_loss(output["final_angles"],
                                    output["final_unnormalized_angles"],
                                    batch["peptide_aatype"],
                                    batch["peptide_self_residues_mask"],
                                    batch["peptide_torsion_angles_mask"][..., 3:],
                                    batch["peptide_torsion_angles_sin_cos"][..., 3:, :],
                                    **openfold_config.loss.supervised_chi)

    # compute fape loss, as in openfold
    fape_losses = _compute_fape_loss(output, batch,
                                     openfold_config.loss.fape)

    # compute violations loss, using an adjusted function
    violation_losses = _compute_cross_violation_loss(output, batch, openfold_config.loss.violation)

    # init total loss at zero
    total_loss = torch.zeros(batch["peptide_aatype"].shape[0], dtype=torch.float, device=batch["peptide_aatype"].device)

    # add fape loss (backbone, sidechain)
    if fape_tune:
        total_loss += 1.0 * fape_losses["total"]

    # add chi loss
    if chi_tune:
        total_loss += 1.0 * chi_loss

    # incorporate affinity loss
    if affinity_tune:
        total_loss += 1.0 * affinity_loss

    # add all fine tune losses (bond lengths, angles, torsions, clashes)
    if fine_tune:
        total_loss += 1.0 * violation_losses["total"]

    # for true non-binders, the total loss is simply affinity-based
    if affinity_tune:
        total_loss[non_binders_index] = 1.0 * affinity_loss[non_binders_index]
    else:
        total_loss[non_binders_index] = 0.0

    # average losses over batch dimension
    result = TensorDict({
        "total": total_loss.mean(dim=0),
        "chi": chi_loss.mean(dim=0),
        "affinity": affinity_loss.mean(dim=0),
    })

    # add these separate components to the result too:
    for component_id, loss_tensor in fape_losses.items():
        result[f"{component_id} fape"] = loss_tensor.mean(dim=0)

    for component_id, loss_tensor in violation_losses.items():
        result[f"{component_id} violation"] = loss_tensor.mean(dim=0)

    return result


def get_calpha_rmsd(output_data: Dict[str, torch.Tensor],
                    batch_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Returns: rmsd per binder id, nonbinders are ignored
    """

    # take binders only
    if "class" in batch_data:
        binders_index = batch_data["class"] == 1

    elif "affinity" in batch_data:
        binders_index = batch_data["affinity"] > AFFINITY_BINDING_TRESHOLD

    else:
        raise ValueError("Cannot compute RMSD without class or affinity output")

    # prevent NaN, in case of no binders
    if not torch.any(binders_index):
        return {}

    ids = [batch_data["ids"][i] for i in torch.nonzero(binders_index)]

    # [n_binders, peptide_maxlen, n_atoms, 3]
    output_positions = output_data["final_positions"][binders_index]
    true_positions = batch_data["peptide_atom14_gt_positions"][binders_index]

    # [n_binders, peptide_maxlen]
    mask = batch_data["peptide_cross_residues_mask"][binders_index]

    # take C-alpha only
    # [n_binders, peptide_maxlen, 3]
    output_positions = output_positions[..., 1, :]
    true_positions = true_positions[..., 1, :]
    squares = (output_positions - true_positions) ** 2

    # [batch_size]
    sum_of_squares = (squares * mask[..., None]).sum(dim=2).sum(dim=1)
    counts = torch.sum(mask.int(), dim=1)

    rmsd = torch.sqrt(sum_of_squares / counts)

    return {ids[i]: rmsd[i].item() for i in range(len(ids))}


def sum_within_peptide_clashes_between_residues(
    output_data: Dict[str, torch.Tensor],
    batch_data: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Returns: sum of clashes between residues within peptide per binder id, nonbinders are ignored
    """

    # take binders only
    if "class" in batch_data:
        binders_index = batch_data["class"] == 1

    elif "affinity" in batch_data:
        binders_index = batch_data["affinity"] > AFFINITY_BINDING_TRESHOLD

    else:
        raise ValueError("Cannot compute RMSD without class or affinity output")

    # prevent NaN, in case of no binders
    if not torch.any(binders_index):
        return {}

    # [n_binders, peptide_maxlen, 14, 3]
    predicted_positions = output_data["final_positions"][binders_index]
    fp_type = predicted_positions.dtype

    # [n_binders, peptide_maxlen, 14]
    atoms_mask = batch_data["peptide_atom14_gt_exists"][binders_index]

    # [n_binders, peptide_maxlen]
    residue_index = batch_data["peptide_residue_index"][binders_index]

    # Compute the Van der Waals radius for every atom
    # (the first letter of the atom name is the element type).
    # [37]
    atomtype_radius = [
        openfold_van_der_waals_radius[name[0]]
        for name in openfold_atom_types
    ]
    # [37]
    atomtype_radius = predicted_positions.new_tensor(atomtype_radius)

    # [n_binders, peptide_maxlen, 14]
    residx_atom14_to_atom37 = batch_data["peptide_residx_atom14_to_atom37"][binders_index]

    # [n_binders, peptide_maxlen, 14]
    atoms_radius = atoms_mask * atomtype_radius[residx_atom14_to_atom37]

    # Create the distance matrix.
    # [n_binders, peptide_maxlen, peptide_maxlen, 14, 14]
    eps = 1e-10
    distances = torch.sqrt(eps +
        torch.sum(
            (predicted_positions[..., :, None, :, None, :] - predicted_positions[..., None, :, None, :, :]) ** 2,
            dim=-1
        )
    )

    # Create the mask for valid distances
    # [n_binders, peptide_maxlen, peptide_maxlen, 14, 14]
    distance_mask = atoms_mask[..., :, None, :, None] * atoms_mask[..., None, :, None, :].type(fp_type)

    # Mask out all the duplicate entries in the lower triangular matrix.
    # Also mask out the diagonal (atom-pairs from the same residue) -- these atoms
    # are handled separately.
    distance_mask = distance_mask * (residue_index[..., :, None, None, None] < residue_index[..., None, :, None, None])

    # Backbone C--N bond between subsequent residues is no clash.
    c_one_hot = torch.nn.functional.one_hot(
        residue_index.new_tensor(2), num_classes=14
    )
    c_one_hot = c_one_hot.reshape(
        *((1,) * len(residue_index.shape[:-1])), *c_one_hot.shape
    )
    c_one_hot = c_one_hot.type(fp_type)

    n_one_hot = torch.nn.functional.one_hot(
        residue_index.new_tensor(0), num_classes=14
    )
    n_one_hot = n_one_hot.reshape(
        *((1,) * len(residue_index.shape[:-1])), *n_one_hot.shape
    )
    n_one_hot = n_one_hot.type(fp_type)

    neighbour_mask = (
        residue_index[..., :, None, None, None] + 1
    ) == residue_index[..., None, :, None, None]

    c_n_bonds = neighbour_mask * c_one_hot[..., None, None, :, None] * n_one_hot[..., None, None, None, :]

    distance_mask = distance_mask * (1.0 - c_n_bonds)

    # Disulfide bridge between two cysteines is no clash.
    cys = openfold_restype_name_to_atom14_names["CYS"]
    cys_sg_idx = cys.index("SG")
    cys_sg_idx = residue_index.new_tensor(cys_sg_idx)
    cys_sg_idx = cys_sg_idx.reshape(*((1,) * len(residue_index.shape[:-1])), 1).squeeze(-1)
    cys_sg_one_hot = torch.nn.functional.one_hot(cys_sg_idx, num_classes=14)
    disulfide_bonds = cys_sg_one_hot[..., None, None, :, None] * cys_sg_one_hot[..., None, None, None, :]
    distance_mask = distance_mask * (1.0 - disulfide_bonds)

    # Compute the lower bound for the allowed distances.
    # [n_binders, peptide_maxlen, peptide_maxlen, 14, 14]
    lower_bounds = distance_mask * (atoms_radius[..., :, None, :, None] + atoms_radius[..., None, :, None, :])

    # [n_binders, peptide_maxlen, peptide_maxlen, 14, 14]
    errors = distance_mask * torch.nn.functional.relu(lower_bounds - distances)

    # [n_binders]
    errors = errors.sum(dim=(-4, -3, -2, -1))

    ids = [batch_data["ids"][i] for i in torch.nonzero(binders_index)]

    return {
        ids[i]: errors[i].item()
        for i in range(len(ids))
    }
