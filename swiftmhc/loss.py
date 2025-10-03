import logging
from math import log
import ml_collections
import torch
from openfold.config import config as openfold_config
from openfold.data.data_transforms import atom37_to_frames as openfold_atom37_to_frames
from openfold.data.data_transforms import make_atom14_masks as openfold_make_atom14_masks
from openfold.np.residue_constants import atom_types as openfold_atom_types
from openfold.np.residue_constants import (
    make_atom14_dists_bounds as openfold_make_atom14_dists_bounds,
)
from openfold.np.residue_constants import (
    restype_atom14_ambiguous_atoms as openfold_restype_atom14_ambiguous_atoms,
)
from openfold.np.residue_constants import (
    restype_name_to_atom14_names as openfold_restype_name_to_atom14_names,
)
from openfold.np.residue_constants import van_der_waals_radius as openfold_van_der_waals_radius
from openfold.utils.loss import between_residue_bond_loss as openfold_between_residue_bond_loss
from openfold.utils.loss import compute_fape as openfold_compute_fape
from openfold.utils.loss import (
    compute_renamed_ground_truth as openfold_compute_renamed_ground_truth,
)
from openfold.utils.loss import sidechain_loss as openfold_compute_sidechain_loss
from openfold.utils.loss import within_residue_violations as openfold_within_residue_violations
from openfold.utils.tensor_utils import batched_gather as openfold_batched_gather
from openfold.utils.tensor_utils import masked_mean as openfold_masked_mean
from swiftmhc.models.model_types import ModelType
from .models.data import TensorDict
from .tools.rigid import Rigid


_log = logging.getLogger(__name__)


def _compute_fape_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: ml_collections.ConfigDict,
) -> dict[str, torch.Tensor]:
    """Compute FAPE loss as in openfold

    Returns:
        backbone:   [*] backbone FAPE
        sidechain:  [*] sidechain FAPE
        total:      [*] backbone FAPE + sidechain FAPE
    """
    # compute backbone FAPE
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

    # Find out which atoms are ambiguous.
    atom14_atom_is_ambiguous = torch.tensor(
        openfold_restype_atom14_ambiguous_atoms[batch["peptide_aatype"].cpu().numpy()],
        device=batch["peptide_aatype"].device,
    )

    renamed_truth = openfold_compute_renamed_ground_truth(
        {
            "atom14_gt_positions": batch["peptide_atom14_gt_positions"],
            "atom14_alt_gt_positions": batch["peptide_atom14_alt_gt_positions"],
            "atom14_gt_exists": batch["peptide_atom14_gt_exists"].float(),
            "atom14_atom_is_ambiguous": atom14_atom_is_ambiguous,
            "atom14_alt_gt_exists": batch["peptide_atom14_gt_exists"].float(),
        },
        output["final_positions"],
    )

    # Get the truth frames and alternative truth frames from the true atom positions,
    # This involves converting from 14-atoms to 37-atoms format.
    peptide_residx_atom37_to_atom14 = batch["peptide_residx_atom37_to_atom14"]
    atom37_positions = openfold_batched_gather(
        batch["peptide_atom14_gt_positions"],
        peptide_residx_atom37_to_atom14,
        dim=-2,
        no_batch_dims=len(batch["peptide_atom14_gt_positions"].shape[:-2]),
    )
    atom37_mask = openfold_batched_gather(
        batch["peptide_atom14_gt_exists"],
        peptide_residx_atom37_to_atom14,
        dim=-1,
        no_batch_dims=len(batch["peptide_atom14_gt_exists"].shape[:-1]),
    )
    truth_frames = openfold_atom37_to_frames(
        {
            "aatype": batch["peptide_aatype"],
            "all_atom_positions": atom37_positions,
            "all_atom_mask": atom37_mask,
        }
    )

    # compute the actual sidechain FAPE
    sc_loss = openfold_compute_sidechain_loss(
        sidechain_frames=output["final_sidechain_frames"][None, ...],
        sidechain_atom_pos=output["final_positions"][None, ...],
        rigidgroups_gt_frames=truth_frames["rigidgroups_gt_frames"],
        rigidgroups_alt_gt_frames=truth_frames["rigidgroups_alt_gt_frames"],
        rigidgroups_gt_exists=truth_frames["rigidgroups_gt_exists"],
        renamed_atom14_gt_positions=renamed_truth["renamed_atom14_gt_positions"],
        renamed_atom14_gt_exists=renamed_truth["renamed_atom14_gt_exists"],
        alt_naming_is_better=renamed_truth["alt_naming_is_better"],
        **config.sidechain,
    )

    total_loss = 0.5 * bb_loss + 0.5 * sc_loss

    return {
        "total": total_loss,
        "backbone": bb_loss,
        "sidechain": sc_loss,
    }


def _between_residue_clash_loss(
    atom14_pred_positions: torch.Tensor,
    atom14_atom_exists: torch.Tensor,
    atom14_atom_radius: torch.Tensor,
    residue_index: torch.Tensor,
    asym_id: torch.Tensor | None = None,
    overlap_tolerance_soft=1.5,
    overlap_tolerance_hard=1.5,
    eps=1e-10,
    max_distance_cutoff=15.0,
    chunk_size: int = 32,
) -> dict[str, torch.Tensor]:
    """Memory-optimized clash loss with distance cutoff.

    Args:
        atom14_pred_positions: Predicted positions of atoms in global prediction frame
        atom14_atom_exists: Mask denoting whether atom at positions exists for given amino acid type
        atom14_atom_radius: Van der Waals radius for each atom
        residue_index: Residue index for given amino acid
        asym_id: Asymmetric unit ID for each residue
        overlap_tolerance_soft: Soft tolerance factor
        overlap_tolerance_hard: Hard tolerance factor
        eps: Small epsilon for numerical stability
        max_distance_cutoff: Maximum distance to consider for clash detection.
            Atom pairs further apart than this are ignored.
        chunk_size: Number of residues to process at once to control memory usage

    Returns:
        Dict containing:
            * 'mean_loss': average clash loss
            * 'per_atom_loss_sum': sum of all clash losses per atom, shape (N, 14)
            * 'per_atom_clash_mask': mask whether atom clashes with any other atom
                shape (N, 14)
            * 'per_atom_num_clash': number of clashes per atom, shape (N, 14)
    """
    fp_type = atom14_pred_positions.dtype
    device = atom14_pred_positions.device
    batch_dims = atom14_pred_positions.shape[:-3]
    n_residues = atom14_pred_positions.shape[-3]

    # Initialize accumulation tensors
    total_loss_sum = torch.zeros((), dtype=fp_type, device=device)
    total_mask_sum = torch.zeros((), dtype=fp_type, device=device)
    per_atom_loss_sum = torch.zeros((*batch_dims, n_residues, 14), dtype=fp_type, device=device)
    per_atom_clash_mask = torch.zeros((*batch_dims, n_residues, 14), dtype=fp_type, device=device)
    per_atom_num_clash = torch.zeros((*batch_dims, n_residues, 14), dtype=fp_type, device=device)

    # Pre-compute residue center of mass for distance-based early termination
    # Shape: (..., N, 3)
    residue_com = torch.sum(atom14_pred_positions * atom14_atom_exists[..., None], dim=-2) / (
        torch.sum(atom14_atom_exists, dim=-1, keepdim=True) + eps
    )

    # Pre-compute one-hot encodings for bond masking
    c_one_hot = torch.nn.functional.one_hot(residue_index.new_tensor(2), num_classes=14)
    c_one_hot = c_one_hot.reshape(*((1,) * len(residue_index.shape[:-1])), *c_one_hot.shape).type(
        fp_type
    )
    n_one_hot = torch.nn.functional.one_hot(residue_index.new_tensor(0), num_classes=14)
    n_one_hot = n_one_hot.reshape(*((1,) * len(residue_index.shape[:-1])), *n_one_hot.shape).type(
        fp_type
    )

    # Pre-compute cysteine disulfide bond mask
    cys = openfold_restype_name_to_atom14_names["CYS"]
    cys_sg_idx = cys.index("SG")
    cys_sg_idx = residue_index.new_tensor(cys_sg_idx)
    cys_sg_idx = cys_sg_idx.reshape(*((1,) * len(residue_index.shape[:-1])), 1).squeeze(-1)
    cys_sg_one_hot = torch.nn.functional.one_hot(cys_sg_idx, num_classes=14)

    # Pre-allocate reusable tensors to reduce memory allocation overhead
    max_chunk = chunk_size
    # Pre-allocate distance tensor (reused across iterations)
    dists_buffer = torch.full(
        (*batch_dims, max_chunk, max_chunk, 14, 14),
        max_distance_cutoff + 1.0,
        dtype=fp_type,
        device=device,
    )

    # Process residues in chunks to avoid O(N²) memory usage
    for i_start in range(0, n_residues, chunk_size):
        i_end = min(i_start + chunk_size, n_residues)
        i_chunk_size = i_end - i_start  # Actual chunk size (might be smaller than chunk_size)

        i_chunk_positions = atom14_pred_positions[..., i_start:i_end, :, :]
        i_chunk_exists = atom14_atom_exists[..., i_start:i_end, :]
        i_chunk_radius = atom14_atom_radius[..., i_start:i_end, :]
        i_chunk_residue_idx = residue_index[..., i_start:i_end]
        i_chunk_asym_id = asym_id[..., i_start:i_end] if asym_id is not None else None
        i_chunk_com = residue_com[..., i_start:i_end, :]

        for j_start in range(i_start, n_residues, chunk_size):  # Only upper triangle
            j_end = min(j_start + chunk_size, n_residues)
            j_chunk_size = j_end - j_start  # Actual chunk size (might be smaller than chunk_size)

            j_chunk_positions = atom14_pred_positions[..., j_start:j_end, :, :]
            j_chunk_exists = atom14_atom_exists[..., j_start:j_end, :]
            j_chunk_radius = atom14_atom_radius[..., j_start:j_end, :]
            j_chunk_residue_idx = residue_index[..., j_start:j_end]
            j_chunk_asym_id = asym_id[..., j_start:j_end] if asym_id is not None else None
            j_chunk_com = residue_com[..., j_start:j_end, :]

            # Early termination: check residue center-of-mass distances
            # Shape: (..., i_chunk_size, j_chunk_size)
            com_distances_sq = torch.sum(
                (i_chunk_com[..., :, None, :] - j_chunk_com[..., None, :, :]) ** 2, dim=-1
            )
            close_residue_pairs = com_distances_sq < (max_distance_cutoff**2)

            # Only consider pairs where i_residue_idx < j_residue_idx (upper triangular)
            residue_pair_mask = (
                i_chunk_residue_idx[..., :, None] < j_chunk_residue_idx[..., None, :]
            )
            close_residue_pairs = close_residue_pairs & residue_pair_mask

            # Reuse pre-allocated buffer and reset for current chunks
            dists_buffer.fill_(max_distance_cutoff + 1.0)
            dists = dists_buffer[..., :i_chunk_size, :j_chunk_size, :, :]

            # Compute distances for all pairs (masked computation)
            close_mask = close_residue_pairs[..., None, None]
            pos_diff = (
                i_chunk_positions[..., :, None, :, None, :]
                - j_chunk_positions[..., None, :, None, :, :]
            )
            # Use torch.linalg.vector_norm for more efficient distance computation
            actual_dists = torch.linalg.vector_norm(pos_diff, dim=-1) + eps

            # Apply masking to distances
            dists = torch.where(close_mask, actual_dists, dists)

            # Create mask for valid atom pairs
            dists_mask = (
                i_chunk_exists[..., :, None, :, None]
                * j_chunk_exists[..., None, :, None, :]
                * close_residue_pairs[..., :, :, None, None]
            ).type(fp_type)

            # Skip C-N bonds between consecutive residues
            # Fix the condition to handle chunk boundaries properly
            if j_start <= i_end:  # Only check when chunks might contain consecutive residues
                neighbour_mask = (i_chunk_residue_idx[..., :, None] + 1) == j_chunk_residue_idx[
                    ..., None, :
                ]
                if i_chunk_asym_id is not None and j_chunk_asym_id is not None:
                    neighbour_mask = neighbour_mask & (
                        i_chunk_asym_id[..., :, None] == j_chunk_asym_id[..., None, :]
                    )

                neighbour_mask = neighbour_mask[..., None, None]
                c_n_bonds = (
                    neighbour_mask
                    * c_one_hot[..., None, None, :, None]
                    * n_one_hot[..., None, None, None, :]
                )
                dists_mask = dists_mask * (1.0 - c_n_bonds)

            # Skip disulfide bonds
            disulfide_bonds = (
                cys_sg_one_hot[..., None, None, :, None] * cys_sg_one_hot[..., None, None, None, :]
            )
            dists_mask = dists_mask * (1.0 - disulfide_bonds)

            # Use masking for torch.compile compatibility
            # Compute distance bounds
            dists_lower_bound = dists_mask * (
                i_chunk_radius[..., :, None, :, None] + j_chunk_radius[..., None, :, None, :]
            )

            # Compute soft clash loss
            dists_to_low_error = dists_mask * torch.nn.functional.relu(
                dists_lower_bound - overlap_tolerance_soft - dists
            )

            # Use in-place operations for memory efficiency
            total_loss_sum.add_(dists_to_low_error.sum())
            total_mask_sum.add_(dists_mask.sum())

            # Accumulate per-atom losses (both directions since we only process upper triangle)
            i_atom_loss = dists_to_low_error.sum(dim=(-3, -1))
            j_atom_loss = dists_to_low_error.sum(dim=(-4, -2))

            per_atom_loss_sum[..., i_start:i_end, :].add_(i_atom_loss)
            per_atom_loss_sum[..., j_start:j_end, :].add_(j_atom_loss)

            # Compute hard clash mask
            clash_mask = dists_mask * (dists < (dists_lower_bound - overlap_tolerance_hard))

            # Accumulate clash counts using in-place operations
            i_clash_count = clash_mask.sum(dim=(-3, -1)).float()  # Sum over j_residues and j_atoms
            j_clash_count = clash_mask.sum(dim=(-4, -2)).float()  # Sum over i_residues and i_atoms

            per_atom_num_clash[..., i_start:i_end, :].add_(i_clash_count)
            per_atom_num_clash[..., j_start:j_end, :].add_(j_clash_count)

            # Update clash mask (any clash detected) using in-place operations
            i_has_clash = torch.amax(clash_mask, dim=(-3, -1)).float()  # Any clash with j_residues
            j_has_clash = torch.amax(clash_mask, dim=(-4, -2)).float()  # Any clash with i_residues

            torch.maximum(
                per_atom_clash_mask[..., i_start:i_end, :],
                i_has_clash,
                out=per_atom_clash_mask[..., i_start:i_end, :],
            )
            torch.maximum(
                per_atom_clash_mask[..., j_start:j_end, :],
                j_has_clash,
                out=per_atom_clash_mask[..., j_start:j_end, :],
            )

    # Compute final mean loss
    mean_loss = total_loss_sum / (1e-6 + total_mask_sum)

    return {
        "mean_loss": mean_loss,  # shape ()
        "per_atom_loss_sum": per_atom_loss_sum,  # shape (N, 14)
        "per_atom_clash_mask": per_atom_clash_mask,  # shape (N, 14)
        "per_atom_num_clash": per_atom_num_clash,  # shape (N, 14)
    }


def _compute_cross_violation_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: ml_collections.ConfigDict,
) -> dict[str, torch.Tensor]:
    """Compute violations in the predicted structure.

    Returns:
        bond:                       [*] bond length violations between residues within the peptide
        CA-C-N-angles:              [*] C-alpha-C-N angle violations in peptide
        C-N-CA-angles:              [*] C-N-C-alpha angle violations in peptide
        between-residues-clash:     [*] clashes between residues from protein and peptide
        within-residues-clash:      [*] clashes between atoms within peptide residues
    """
    # Compute the between residue clash loss. (include both peptide and protein)
    # [*, peptide_maxlen + protein_maxlen, 14]
    residx_atom14_to_atom37 = torch.cat(
        (batch["peptide_residx_atom14_to_atom37"], batch["protein_residx_atom14_to_atom37"]), dim=1
    )

    # [*, peptide_maxlen + protein_maxlen, 14, 3]
    atom14_pred_positions = torch.cat(
        (output["final_positions"], batch["protein_atom14_gt_positions"]), dim=1
    )

    # [*, peptide_maxlen + protein_maxlen, 14]
    atom14_atom_exists = torch.cat(
        (batch["peptide_atom14_gt_exists"], batch["protein_atom14_gt_exists"]), dim=1
    )

    # Compute the Van der Waals radius for every atom
    # (the first letter of the atom name is the element type).
    # [37]
    atomtype_radius = [openfold_van_der_waals_radius[name[0]] for name in openfold_atom_types]
    # [37]
    atomtype_radius = atom14_pred_positions.new_tensor(atomtype_radius)

    # [*, peptide_maxlen + protein_maxlen, 14]
    atom14_atom_radius = atom14_atom_exists * atomtype_radius[residx_atom14_to_atom37]

    # [*, peptide_maxlen]
    peptide_residue_index = batch["peptide_residue_index"]
    # [*, protein_maxlen]
    protein_residue_index = batch["protein_residue_index"]

    # [*, peptide_maxlen + protein_maxlen]
    residue_index = torch.cat((peptide_residue_index, protein_residue_index), dim=1)

    between_residue_clashes = _between_residue_clash_loss(
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

    # [*]
    violations_between_residues_bonds_c_n_loss_mean = connection_violations["c_n_loss_mean"]
    violations_between_residues_angles_ca_c_n_loss_mean = connection_violations["ca_c_n_loss_mean"]
    violations_between_residues_angles_c_n_ca_loss_mean = connection_violations["c_n_ca_loss_mean"]

    # [*, peptide_len + protein_len, 14]
    violations_between_residues_clashes_per_atom_loss_sum = between_residue_clashes[
        "per_atom_loss_sum"
    ]

    # [*, peptide_len, 14]
    violations_within_residues_per_atom_loss_sum = residue_violations["per_atom_loss_sum"]

    # Calculate loss, as in openfold
    peptide_num_atoms = torch.sum(batch["peptide_atom14_gt_exists"])

    # [*]
    between_residues_clash = torch.sum(violations_between_residues_clashes_per_atom_loss_sum) / (
        config.eps + peptide_num_atoms
    )
    within_residues_clash = torch.sum(violations_within_residues_per_atom_loss_sum) / (
        config.eps + peptide_num_atoms
    )

    # [*]
    loss = {
        "bond": violations_between_residues_bonds_c_n_loss_mean,
        "CA-C-N-angles": violations_between_residues_angles_ca_c_n_loss_mean,
        "C-N-CA-angles": violations_between_residues_angles_c_n_ca_loss_mean,
        "between-residues-clash": between_residues_clash,
        "within-residues-clash": within_residues_clash,
        "total": (
            violations_between_residues_bonds_c_n_loss_mean
            + violations_between_residues_angles_ca_c_n_loss_mean
            + violations_between_residues_angles_c_n_ca_loss_mean
            + between_residues_clash
            + within_residues_clash
        ),
    }

    return loss


def _compute_torsion_angle_loss(
    a: torch.Tensor,
    a_mask: torch.Tensor,
    a_gt: torch.Tensor,
    a_alt_gt: torch.Tensor,
) -> torch.Tensor:
    """Torsion angle loss, according to alphafold Algorithm 27
    This code was copied from openfold and modified.
    The original torsion_angle_loss function is at:
    https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/loss.py

    Args:
        a:          [*, N, 7, 2] predicted torsion angles sin, cos
        a_mask:     [*, N, 7] (bool) torsion angles mask
        a_gt:       [*, N, 7, 2] true torsion angles sin, cos
        a_alt_gt:   [*, N, 7, 2] true alternative torsion angles sin, cos

    Returns:
        [*] losses per case
    """
    # [*, N, 7]
    norm = torch.norm(a, dim=-1)

    # [*, N, 7, 2]
    a = a / norm.unsqueeze(-1)

    # [*, N, 7]
    diff_norm_gt = torch.norm(a - a_gt, dim=-1)
    diff_norm_alt_gt = torch.norm(a - a_alt_gt, dim=-1)
    min_diff = torch.minimum(diff_norm_gt**2, diff_norm_alt_gt**2)

    # [*]
    l_torsion = openfold_masked_mean(a_mask, min_diff, dim=(-2, -1))
    l_angle_norm = openfold_masked_mean(a_mask, torch.abs(norm - 1), dim=(-2, -1))

    an_weight = 0.02
    return l_torsion + an_weight * l_angle_norm


# above this value, BA output is considered binding
AFFINITY_BINDING_TRESHOLD = 1.0 - log(500) / log(50000)

# BA loss functions
_classification_loss_function = torch.nn.CrossEntropyLoss(reduction="none")
_regression_loss_function = torch.nn.MSELoss(reduction="none")


def get_loss(
    model_type: ModelType,
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    affinity_tune: bool,
    fape_tune: bool,
    torsion_tune: bool,
    fine_tune: bool,
) -> TensorDict:
    """Compute all losses and sum them up, according to what is desired.

    Args:
        model_type: CLASSIFICATION or REGRESSION
        output: what came out of the model
        batch: what came out of the dataset
        affinity_tune: whether or not to include the BA loss term
        fape_tune: whether or not to include the FAPE loss term
        torsion_tune: whether or not to include the torsion loss term
        fine_tune: whether or not to include bond length, bond angle violations or clashes in the loss

    Returns:
        total:                      [*] the sum of selected loss terms, per batch entry
        torsion:                    [*] torsion loss, per batch entry
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
    torsion_loss = None
    violation_losses = None
    fape_losses = None

    # compute our own affinity-based loss
    affinity_loss = None
    non_binders_index = None
    if model_type == ModelType.REGRESSION:
        if "affinity" in batch:
            affinity_loss = _regression_loss_function(
                output["affinity"].float(), batch["affinity"].float()
            )

            # handle inequalities
            # if truth is t < 0.5 and prediction is t = 0.49, then the loss must be zero
            affinity_loss[
                torch.logical_and(output["affinity"] < batch["affinity"], batch["affinity_lt"])
            ] = 0.0
            affinity_loss[
                torch.logical_and(output["affinity"] > batch["affinity"], batch["affinity_gt"])
            ] = 0.0

            non_binders_index = batch["affinity"] < AFFINITY_BINDING_TRESHOLD

        elif "class" in batch and not affinity_tune:
            # needed for structural loss
            non_binders_index = torch.logical_not(batch["class"])
        else:
            raise ValueError("no affinity data to determine loss")

    elif model_type == ModelType.CLASSIFICATION:
        affinity_loss = _classification_loss_function(
            output["logits"].float(), batch["class"].float()
        )
        non_binders_index = torch.logical_not(batch["class"])
    else:
        raise TypeError(f"unknown model type {model_type}")

    # mapping 14-atoms to 37-atoms format, because several openfold functions use the 37 format
    # required by fape_loss and violation_loss
    peptide_data = openfold_make_atom14_masks({"aatype": batch["peptide_aatype"]})
    protein_data = openfold_make_atom14_masks({"aatype": batch["protein_aatype"]})
    batch["peptide_residx_atom37_to_atom14"] = peptide_data["residx_atom37_to_atom14"]
    batch["protein_residx_atom37_to_atom14"] = protein_data["residx_atom37_to_atom14"]
    batch["peptide_residx_atom14_to_atom37"] = peptide_data["residx_atom14_to_atom37"]
    batch["protein_residx_atom14_to_atom37"] = protein_data["residx_atom14_to_atom37"]

    # init total loss at zero
    total_loss = torch.zeros(
        batch["peptide_aatype"].shape[0],
        dtype=output["final_angles"].dtype,
        device=batch["peptide_aatype"].device,
    )

    # compute fape loss (backbone, sidechain), as in openfold
    if fape_tune:
        fape_losses = _compute_fape_loss(output, batch, openfold_config.loss.fape)
        total_loss += 1.0 * fape_losses["total"]

    # compute torsion loss
    if torsion_tune:
        torsion_loss = _compute_torsion_angle_loss(
            output["final_angles"],
            batch["peptide_torsion_angles_mask"],
            batch["peptide_torsion_angles_sin_cos"],
            batch["peptide_alt_torsion_angles_sin_cos"],
        )
        total_loss += 1.0 * torsion_loss

    # add all fine tune losses (bond lengths, angles, torsions, clashes)
    if fine_tune:
        # compute violations loss, using an adjusted function
        violation_losses = _compute_cross_violation_loss(
            output, batch, openfold_config.loss.violation
        )
        total_loss += 1.0 * violation_losses["total"]

    # add affinity loss
    if affinity_tune:
        total_loss += 1.0 * affinity_loss
        total_loss[non_binders_index] = 1.0 * affinity_loss[non_binders_index]
    else:
        # for true non-binders, the total loss is simply affinity-based
        total_loss[non_binders_index] = 0.0

    # average losses over batch dimension
    result = TensorDict(
        {
            "total": total_loss.mean(dim=0),
        }
    )
    # add these separate components to the result too:
    if fape_losses is not None:
        for component_id, loss_tensor in fape_losses.items():
            result[f"{component_id} fape"] = loss_tensor.mean(dim=0)

    if torsion_loss is not None:
        result["torsion"] = torsion_loss.mean(dim=0)

    if violation_losses is not None:
        for component_id, loss_tensor in violation_losses.items():
            result[f"{component_id} violation"] = loss_tensor.mean(dim=0)

    if affinity_loss is not None:
        result["affinity"] = affinity_loss.mean(dim=0)

    for key in result:
        if "total" not in key and result[key].isnan():
            raise RuntimeError(f"NaN {key} loss")

    return result


def get_calpha_rmsd(
    output_data: dict[str, torch.Tensor], batch_data: dict[str, torch.Tensor]
) -> dict[str, float]:
    """Returns: rmsd per binder id, nonbinders are ignored"""
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
