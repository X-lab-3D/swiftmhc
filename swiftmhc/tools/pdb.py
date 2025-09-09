import logging
from typing import List
from typing import Tuple
import numpy
import torch
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure
from openfold.np.residue_constants import restype_1to3
from openfold.np.residue_constants import (
    restype_name_to_atom14_names as openfold_residue_atom14_names,
)
from openfold.np.residue_constants import restypes


_log = logging.getLogger(__name__)


def _has_atom(residue: Residue, name: str) -> bool:
    """Check whether an atom is present"""
    for atom in residue.get_atoms():
        if atom.name == name:
            return True

    return False


def _get_atom(residue: Residue, name: str) -> Atom:
    """Looks up the atom by name"""
    if name == "":
        return None

    for atom in residue.get_atoms():
        if atom.name == name:
            return atom

    raise ValueError(f"{residue} has no such atom: {name}")


def get_atom14_positions(residue: Residue) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get the positions of the atoms for one residue.

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
        if len(atom_name) > 0 and _has_atom(residue, atom_name):
            atom = _get_atom(residue, atom_name)
            positions.append(atom.coord)
            masks.append(True)
        else:
            masks.append(False)
            positions.append((0.0, 0.0, 0.0))

    return torch.tensor(numpy.array(positions)), torch.tensor(masks)


amino_acid_order = [restype_1to3[letter] for letter in restypes]


def recreate_structure(
    structure_id: str,
    data_by_chain: List[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Structure:
    """Recreate a PDB structure from swiftmhc model output data.

    Args:
        structure_id, to be used in the structure
        data_by_chain:
            [0] chain ids
            [1] residue numbers
            [2] amino acids per residue indexes
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
    for chain_id, residue_numbers, aatype, atom_positions, atom_exists in data_by_chain:
        chain = Chain(chain_id)
        model.add(chain)

        # iter residues in chain
        for residue_index, amino_acid_index in enumerate(aatype):
            residue_name = amino_acid_order[amino_acid_index]
            residue_number = residue_numbers[residue_index]

            residue = Residue((" ", residue_number, " "), residue_name, chain_id)

            # iter atoms in residue
            for atom_index, atom_name in enumerate(openfold_residue_atom14_names[residue_name]):
                if not atom_exists[residue_index][atom_index] or len(atom_name) == 0:
                    continue

                position = atom_positions[residue_index][atom_index]

                atom_count += 1
                atom = Atom(atom_name, position, 0.0, 1.0, " ", atom_name, atom_count, atom_name[0])

                residue.add(atom)

            chain.add(residue)

    return structure
