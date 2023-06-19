from typing import List, Tuple, Union, Optional, Dict
import os
import logging
from math import isinf

import h5py
import pandas
from pdb2sql import pdb2sql
import numpy
import torch
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from blosum import BLOSUM

from openfold.np.residue_constants import restype_atom37_mask
from openfold.data.data_transforms import (atom37_to_frames,
                                           atom37_to_torsion_angles,
                                           get_backbone_frames,
                                           make_atom14_masks,
                                           make_atom14_positions)
from openfold.utils.feats import atom14_to_atom37

from .tools.pdb import (get_residue_transformations,
                        get_atom14_positions,
                        generate_symmetry_alternative,
                        get_residue_proximities)
from .domain.amino_acid import amino_acids_by_code, canonical_amino_acids
from .models.amino_acid import AminoAcid


_log = logging.getLogger(__name__)


PREPROCESS_KD_NAME = "kd"
PREPROCESS_PROTEIN_NAME = "protein"
PREPROCESS_LOOP_NAME = "loop"


def _write_preprocessed_data(hdf5_path: str, storage_id: str,
                             protein_data: Dict[str, torch.Tensor],
                             loop_data: Dict[str, torch.Tensor],
                             proximities: torch.Tensor,
                             kd: Optional[float] = None):
    """
    Args:
        protein_proximities: [protein_len, protein_len, 1] proximity matrix
        proximities:         [loop_len, protein_len, 1] proximity matrix
    """

    with h5py.File(hdf5_path, 'a') as hdf5_file:

        storage_group = hdf5_file.require_group(storage_id)

        if kd is not None:
            storage_group.create_dataset(PREPROCESS_KD_NAME, data=kd)

        storage_group.create_dataset("proximities", data=proximities, compression="lzf")

        protein_group = storage_group.require_group(PREPROCESS_PROTEIN_NAME)
        for field_name, field_data in protein_data.items():
            protein_group.create_dataset(field_name, data=field_data, compression="lzf")

        loop_group = storage_group.require_group(PREPROCESS_LOOP_NAME)
        for field_name, field_data in loop_data.items():
            loop_group.create_dataset(field_name, data=field_data, compression="lzf")


def _read_affinities_by_id(table_path: str) -> List[Tuple[str, float]]:
    """
    Args:
        table_path: points to a csv table
    """

    table = pandas.read_csv(table_path)

    affinities_by_id = []
    for index, row in table.iterrows():
        kd = row["measurement_value"]
        id_ = row["ID"]

        affinities_by_id.append((id_, kd))

    return affinities_by_id


def _read_mask_data(path: str) -> List[Tuple[str, int, AminoAcid]]:

    mask_data = []

    with open(path, 'r') as f:
        for line in f:
            if not line.startswith("#"):
                chain_id, resnum_s, amino_acid_s = line.split()

                residue_number = int(resnum_s)
                amino_acid = amino_acids_by_code[amino_acid_s]

                mask_data.append((chain_id, residue_number, amino_acid))

    return mask_data


def _get_blosum_encoding(amino_acid_indexes: List[int], blosum_index: int) -> List[int]:
    """
    Arguments:
        amino_acid_indexes: order of numbers 0 to 19, coding for the amino acids
        blosum_index: identifies the type of BLOSUM matrix to use
    Returns:
        the amino acids encoded by their BLOSUM rows
    """

    matrix = BLOSUM(blosum_index)
    encoding = []
    for amino_acid_index in amino_acid_indexes:
        amino_acid = canonical_amino_acids[amino_acid_index]

        row = []
        for other_amino_acid in canonical_amino_acids:

            if isinf(matrix[amino_acid.one_letter_code][other_amino_acid.one_letter_code]):

                raise ValueError(f"not found in blosum matrix: {amino_acid.one_letter_code} & {other_amino_acid.one_letter_code}")
            else:
                row.append(matrix[amino_acid.one_letter_code][other_amino_acid.one_letter_code])

        encoding.append(row)

    return torch.tensor(encoding)


def _mask_residues(residues: List[Residue], mask_ids: List[Tuple[str, int, AminoAcid]]) -> torch.Tensor:

    mask = []
    for residue in residues:

        full_id = residue.get_full_id()
        if len(full_id) == 4:
            structure_id, model_id, chain_id, residue_id = full_id
        else:
            chain_id, residue_id = full_id

        residue_number = residue_id[1]
        amino_acid = amino_acids_by_code[residue.get_resname()]

        residue_id = (chain_id, residue_number, amino_acid)

        mask.append(residue_id in mask_ids)

    mask = torch.tensor(mask, dtype=torch.bool)

    if not torch.any(mask):
        raise ValueError(f"none found of {mask_ids}")

    return mask


def _read_residue_data(residues: List[Residue]) -> Dict[str, torch.Tensor]:
    """
    Returns:
        aatype: [len] sequence, indices of amino acids
        sequence_onehot: [len, depth] sequence, one-hot encoded amino acids
        backbone_rigid_tensor: [len, 4, 4] 4x4 representation of the backbone frames
        torsion_angles_sin_cos: [len, 7, 2]
        alt_torsion_angles_sin_cos: [len, 7, 2]
        torsion_angles_mask: [len, 7]
        atom14_gt_exists: [len, 14]
        atom14_gt_positions: [len, 14, 3]
        atom14_alt_gt_positions: [len, 14, 3]
        residx_atom14_to_atom37: [len, 14]
    """

    if len(residues) < 3:
        raise ValueError(f"Only {len(residues)} residues")

    # embed the sequence
    amino_acids = [amino_acids_by_code[r.get_resname()] for r in residues]
    sequence_onehot = torch.stack([aa.one_hot_code for aa in amino_acids])
    aatype = torch.tensor([aa.index for aa in amino_acids])

    # get atom positions and mask
    atom14_positions = []
    atom14_mask = []
    for residue_index, residue in enumerate(residues):
        p, m = get_atom14_positions(residue)
        atom14_positions.append(p.float())
        atom14_mask.append(m)
    atom14_positions = torch.stack(atom14_positions)
    atom14_mask = torch.stack(atom14_mask)

    blosum62 = _get_blosum_encoding(aatype, 62)

    # convert to atom 37 format, for the frames and torsion angles
    protein = {
        "aatype": aatype,
        "sequence_onehot": sequence_onehot,
        "blosum62": blosum62
    }
    protein = make_atom14_masks(protein)

    atom37_positions = atom14_to_atom37(atom14_positions, protein)
    protein["all_atom_mask"] = protein["atom37_atom_exists"]
    protein["all_atom_positions"] = atom37_positions

    # get frames, torsion angles and alternative positions
    protein = atom37_to_frames(protein)
    protein = atom37_to_torsion_angles("")(protein)
    protein = get_backbone_frames(protein)
    protein = make_atom14_positions(protein)

    return protein


def _create_symmetry_alternative(chain: Chain) -> Chain:
    alt_chain = Chain(chain.id)

    for residue in chain.get_residues():
        alt_chain.add(generate_symmetry_alternative(residue))

    return alt_chain


def _create_proximities(residues1: List[Residue], residues2: List[Residue]) -> torch.Tensor:

    residue_distances = torch.empty((len(residues1), len(residues2), 1), dtype=torch.float32)

    for i in range(len(residues1)):

        atoms_i = list(residues1[i].get_atoms())
        atom_count_i = len(atoms_i)
        atom_positions_i = torch.tensor(numpy.array([atom.coord for atom in atoms_i]))

        for j in range(len(residues2)):

            atoms_j = list(residues2[j].get_atoms())
            atom_count_j = len(atoms_j)
            atom_positions_j = torch.tensor(numpy.array([atom.coord for atom in atoms_j]))

            atomic_distances_ij = torch.cdist(atom_positions_i, atom_positions_j, p=2)

            min_distance = torch.min(atomic_distances_ij).item()

            residue_distances[i, j, 0] = min_distance

    return 1.0 / (1.0 + residue_distances)


def preprocess(table_path: str,
               models_path: str,
               protein_self_mask_path: str,
               protein_cross_mask_path: str,
               output_path: str):

    affinities_by_id = _read_affinities_by_id(table_path)

    protein_residues_self_mask = _read_mask_data(protein_self_mask_path)
    protein_residues_cross_mask = _read_mask_data(protein_cross_mask_path)

    for id_, kd in affinities_by_id:

        # parse the pdb file
        model_path = os.path.join(models_path, f"{id_}.pdb")

        pdb_parser = PDBParser()

        structure = pdb_parser.get_structure(id_, model_path)

        # locate protein and loop
        chains_by_id = {c.id: c for c in structure.get_chains()}
        if "M" not in chains_by_id:
            raise ValueError(f"missing protein chain M in {model_path}")
        if "P" not in chains_by_id:
            raise ValueError(f"missing loop chain P in {model_path}")

        # get residues from the protein (chain M)
        protein_chain = chains_by_id["M"]
        protein_residues = list(protein_chain.get_residues())

        # determine which proteinresidues match with the mask
        self_residues_mask = _mask_residues(protein_residues, protein_residues_self_mask)
        cross_residues_mask = _mask_residues(protein_residues, protein_residues_cross_mask)

        # derive data from protein residues
        protein_data = _read_residue_data(protein_residues)

        # remove the residues that are completely masked out.
        # (false in both masks)
        combo_mask = torch.logical_or(self_residues_mask, cross_residues_mask)
        combo_mask_nonzero = combo_mask.nonzero()
        mask_start = combo_mask_nonzero.min()
        mask_end = combo_mask_nonzero.max() + 1

        for key, value in protein_data.items():
            protein_data[key] = value[mask_start: mask_end, ...]

        # store masks as protein data
        protein_data["self_residues_mask"] = self_residues_mask[mask_start: mask_end]
        protein_data["cross_residues_mask"] = cross_residues_mask[mask_start: mask_end]

        # get residues from the loop (chain P)
        loop_chain = chains_by_id["P"]
        loop_residues = list(loop_chain.get_residues())
        loop_data = _read_residue_data(loop_residues)

        # proximities between loop and protein
        proximities = _create_proximities(loop_residues, protein_residues)

        # proximities within protein
        protein_data["proximities"] = _create_proximities(protein_residues, protein_residues)

        _write_preprocessed_data(output_path, id_,
                                 protein_data,
                                 loop_data,
                                 proximities, kd)
