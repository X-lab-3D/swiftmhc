from typing import List, Tuple
from collections import OrderedDict
from math import sqrt
import logging

import numpy
import torch
from torch.nn.functional import normalize

from openfold.utils.rigid_utils import Rigid
from openfold.np.residue_constants import (restype_name_to_atom14_names as openfold_residue_atom14_names,
                                           chi_angles_atoms as openfold_chi_angles_atoms,
                                           chi_angles_mask as openfold_chi_angles_mask)
from Bio.PDB.vectors import Vector, calc_dihedral
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
from Bio.PDB.Structure import Structure
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model

from ..domain.amino_acid import amino_acids_by_code
from .amino_acid import one_hot_decode_sequence


_log = logging.getLogger(__name__)


def _get_atom(residue: Residue, name: str) -> Atom:
    "looks up the atom by name"

    if name == "":
        return None

    for atom in residue.get_atoms():
        if atom.name == name:
            return atom

    raise ValueError(f"{residue} has no such atom: {name}")


def get_atom14_positions(residue: Residue) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get the positions of the atoms for one residue.

    Args:
        residue: the residue to get the atoms for
    Returns:
        the residue's atom positions in openfold atom14 format
        the masks for the atoms in openfold atom14 format
    """

    atom_names = openfold_residue_atom14_names[residue.get_resname()]

    masks = []
    positions = []
    for atom_name in atom_names:

        if len(atom_name) > 0:
            atom = _get_atom(residue, atom_name)
            positions.append(atom.coord)
            masks.append(True)
        else:
            masks.append(False)
            positions.append((0.0, 0.0, 0.0))

    return torch.tensor(numpy.array(positions)), torch.tensor(masks)


def recreate_structure(structure_id: str,
                       data_by_chain: List[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) -> Structure:
    """
        Recreate a PDB structure from swiftmhc model output data.

        Args:
            structure_id, to be used in the structure
            data_by_chain:
                [0] chain ids
                [1] residue numbers
                [2] amino acids per residue, one-hot encoded
                [3] positions of atoms: [n_residues, n_atoms, 3]
                [4] whether the atoms exist or not [n_residues, n_atoms]
        Returns:
            the resulting structure
    """

    # create structure with given id and single model
    structure = Structure(structure_id)
    model = Model(0)
    structure.add(model)

    atom_count = 0

    # iter chains
    for chain_id, residue_numbers, sequence, atom_positions, atom_exists in data_by_chain:

        chain = Chain(chain_id)
        model.add(chain)

        amino_acids = one_hot_decode_sequence(sequence)

        # iter residues in chain
        for residue_index, amino_acid in enumerate(amino_acids):

            if amino_acid is None:
                continue

            residue_name = amino_acid.three_letter_code
            residue_number = residue_numbers[residue_index]

            residue = Residue((" ", residue_number, " "),
                              residue_name, chain_id)

            # iter atoms in residue
            for atom_index, atom_name in enumerate(openfold_residue_atom14_names[residue_name]):

                if not atom_exists[residue_index][atom_index].item():
                    continue

                position = atom_positions[residue_index][atom_index]

                atom_count += 1
                atom = Atom(atom_name, position, 0.0, 1.0, " ", atom_name,
                            atom_count, atom_name[0])

                residue.add(atom)

            chain.add(residue)

    return structure


