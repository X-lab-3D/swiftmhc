from typing import List, Tuple, Union, Optional, Dict
import os
import logging
from math import isinf

import h5py
import pandas
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
from .models.complex import ComplexClass


_log = logging.getLogger(__name__)


PREPROCESS_KD_NAME = "kd"
PREPROCESS_CLASS_NAME = "class"
PREPROCESS_PROTEIN_NAME = "protein"
PREPROCESS_LOOP_NAME = "loop"


def _write_preprocessed_data(hdf5_path: str, storage_id: str,
                             protein_data: Dict[str, torch.Tensor],
                             loop_data: Dict[str, torch.Tensor],
                             target: Optional[Union[float, ComplexClass]] = None):

    with h5py.File(hdf5_path, 'a') as hdf5_file:

        storage_group = hdf5_file.require_group(storage_id)

        if isinstance(target, float):
            storage_group.create_dataset(PREPROCESS_KD_NAME, data=kd)

        elif isinstance(target, ComplexClass):
            storage_group.create_dataset(PREPROCESS_CLASS_NAME, data=int(target))
        else:
            raise TypeError(type(target))

        protein_group = storage_group.require_group(PREPROCESS_PROTEIN_NAME)
        for field_name, field_data in protein_data.items():
            protein_group.create_dataset(field_name, data=field_data, compression="lzf")

        loop_group = storage_group.require_group(PREPROCESS_LOOP_NAME)
        for field_name, field_data in loop_data.items():
            loop_group.create_dataset(field_name, data=field_data, compression="lzf")


def _read_targets_by_id(table_path: str) -> List[Tuple[str, Union[float, ComplexClass]]]:
    """
    Args:
        table_path: points to a csv table
    """

    table = pandas.read_csv(table_path)

    data = []
    for index, row in table.iterrows():
        value = row["measurement_value"]
        id_ = row["ID"]

        try:
            value = float(value)
        except ValueError:
            value = ComplexClass.from_string(value)

        data.append((id_, value))

    return data


def _read_mask_data(path: str) -> List[Tuple[str, int, AminoAcid]]:

    mask_data = []

    with open(path, 'r') as f:
        for line in f:
            if not line.startswith("#"):
                row = line.split()

                chain_id = row[0]
                residue_number = int(row[1])

                mask_data.append((chain_id, residue_number))

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


def _mask_residues(residues: List[Residue], mask_ids: List[Tuple[str, int]]) -> torch.Tensor:

    mask = []
    for residue in residues:

        full_id = residue.get_full_id()
        if len(full_id) == 4:
            structure_id, model_id, chain_id, residue_id = full_id
        else:
            chain_id, residue_id = full_id

        residue_number = residue_id[1]

        residue_id = (chain_id, residue_number)

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
    residue_numbers = []
    for residue_index, residue in enumerate(residues):
        p, m = get_atom14_positions(residue)
        atom14_positions.append(p.float())
        atom14_mask.append(m)
        residue_numbers.append(residue.get_id()[1])

    atom14_positions = torch.stack(atom14_positions)
    atom14_mask = torch.stack(atom14_mask)
    residue_numbers = torch.tensor(residue_numbers)

    blosum62 = _get_blosum_encoding(aatype, 62)

    # convert to atom 37 format, for the frames and torsion angles
    protein = {
        "residue_numbers": residue_numbers,
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

    targets_by_id = _read_targets_by_id(table_path)

    protein_residues_self_mask = _read_mask_data(protein_self_mask_path)
    protein_residues_cross_mask = _read_mask_data(protein_cross_mask_path)

    for id_, target in targets_by_id:

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

        try:
            # determine which proteinresidues match with the mask
            self_residues_mask = _mask_residues(protein_residues, protein_residues_self_mask)
            cross_residues_mask = _mask_residues(protein_residues, protein_residues_cross_mask)

            # remove the residues that are completely outside of mask range
            combo_mask = torch.logical_or(self_residues_mask, cross_residues_mask)
            combo_mask_nonzero = combo_mask.nonzero()
            mask_start = combo_mask_nonzero.min()
            mask_end = combo_mask_nonzero.max() + 1

            # apply the limiting protein range, reducing the size of the data that needs to be generated.
            self_residues_mask = self_residues_mask[mask_start: mask_end]
            cross_residues_mask = cross_residues_mask[mask_start: mask_end]
            protein_residues = protein_residues[mask_start: mask_end]

            # derive data from protein residues
            protein_data = _read_residue_data(protein_residues)
            protein_data["cross_residues_mask"] = cross_residues_mask
            protein_data["self_residues_mask"] = self_residues_mask

            # get residues from the loop (chain P)
            loop_chain = chains_by_id["P"]
            loop_residues = list(loop_chain.get_residues())
            loop_data = _read_residue_data(loop_residues)

            # proximities within protein
            protein_proximities = _create_proximities(protein_residues, protein_residues)
            protein_data["proximities"] = protein_proximities

            _write_preprocessed_data(output_path, id_,
                                     protein_data,
                                     loop_data,
                                     target)
        except:
            _log.exception(f"on {model_path}")
            continue
