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
                             distances: torch.Tensor,
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

        storage_group.create_dataset("distances", data=distances, compression="lzf")

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


def _get_masked_residues(id_: str,
                         residues: List[Residue],
                         mask: List[Tuple[str, int, AminoAcid]]) -> List[Residue]:
    passed = []
    for residue in residues:
        full_id = residue.get_full_id()
        if len(full_id) == 4:
            structure_id, model_id, chain_id, residue_id = full_id
        else:
            chain_id, residue_id = full_id

        residue_number = residue_id[1]
        amino_acid = amino_acids_by_code[residue.get_resname()]

        t = (chain_id,
             residue_number,
             amino_acid)

        for mask_chain, mask_resnum, mask_aa in mask:
            if mask_chain == chain_id and mask_resnum == residue_number:

                if mask_aa != amino_acid:
                    raise ValueError(f"at {id_} {chain_id} {residue_number}: expected {mask_aa}, but found {amino_acid}")

                passed.append(residue)

    return passed


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

    return encoding


def _read_residue_data(id_: str,
                       chain: Chain,
                       residue_mask: Optional[List[Tuple[str, int, AminoAcid]]] = None,
                       ) -> Union[Tuple[torch.tensor, torch.tensor, torch.tensor],
                                  Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]]:
    """
    Arguments:
        residue_mask:
            which residues to include, consists of elements, each containing chain id, residue number, amino acid
    Returns:
        aatype: [len] sequence, indices of amino acids
        sequence_embedding: [len, depth] sequence, one-hot encoded amino acids
        backbone_rigid_tensor: [len, 4, 4] 4x4 representation of the backbone frames
        torsion_angles_sin_cos: [len, 7, 2]
        alt_torsion_angles_sin_cos: [len, 7, 2]
        torsion_angles_mask: [len, 7]
        atom14_gt_exists: [len, 14]
        atom14_gt_positions: [len, 14, 3]
        atom14_alt_gt_positions: [len, 14, 3]
        residx_atom14_to_atom37: [len, 14]
    """

    # take the residues of the chain, that are mentioned in the mask (if any)
    residues = list(chain.get_residues())
    if residue_mask is not None:
        residues = _get_masked_residues(id_, residues, residue_mask)

    if len(residues) < 3:
        raise ValueError(f"found only residues {residues} in {chain.id}, using mask {residue_mask}")

    # embed the sequence
    amino_acids = [amino_acids_by_code[r.get_resname()] for r in residues]
    sequence_embedding = torch.stack([aa.one_hot_code for aa in amino_acids])
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
    protein = {"aatype": aatype,
               "sequence_embedding": sequence_embedding,
               "blosum62": blosum62}
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


def _create_distances(id_: str,
                      chain1: Chain, chain2: Chain,
                      chain1_mask: Optional[List[Tuple[str, int, AminoAcid]]] = None,
                      chain2_mask: Optional[List[Tuple[str, int, AminoAcid]]] = None) -> torch.Tensor:

    residues1 = list(chain1.get_residues())
    if chain1_mask is not None:
        residues1 = _get_masked_residues(id_, residues1, chain1_mask)

    residues2 = list(chain2.get_residues())
    if chain2_mask is not None:
        residues2 = _get_masked_residues(id_, residues2, chain2_mask)

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

    return residue_distances


def preprocess(table_path: str, models_path: str, mask_path: str, output_path: str):

    affinities_by_id = _read_affinities_by_id(table_path)

    protein_mask = _read_mask_data(mask_path)

    for id_, kd in affinities_by_id:

        try:
            model_path = os.path.join(models_path, f"{id_}.pdb")

            pdb_parser = PDBParser()

            structure = pdb_parser.get_structure(id_, model_path)

            protein_chain = list(filter(lambda c: c.id == "M", structure.get_chains()))[0]
            protein_data = _read_residue_data(id_, protein_chain, residue_mask=protein_mask)

            loop_chain = list(filter(lambda c: c.id == "P", structure.get_chains()))[0]
            loop_data = _read_residue_data(id_, loop_chain)

            distances = _create_distances(id_, loop_chain, protein_chain, None, protein_mask)
            protein_data["distances"] = _create_distances(id_, protein_chain, protein_chain, protein_mask, protein_mask)

            _write_preprocessed_data(output_path, id_,
                                     protein_data,
                                     loop_data,
                                     distances, kd)
        except:
            _log.exception(f"cannot preprocess {id_}")
