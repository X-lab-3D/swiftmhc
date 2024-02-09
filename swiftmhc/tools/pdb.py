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

    if name == "":
        return None

    for atom in residue.get_atoms():
        if atom.name == name:
            return atom

    raise ValueError(f"{residue} has no such atom: {name}")


def _dup_residue_but_swap_atoms(input_residue: Residue,
                                atom_names_to_swap: List[Tuple[str, str]]) -> Residue:

    output_residue = Residue(input_residue.id, input_residue.resname, input_residue.segid)
    input_atoms_by_name = {atom.name: atom for atom in input_residue.get_atoms()}

    swap_atom_list = []
    for n1, n2 in atom_names_to_swap:
        swap_atom_list += [n1, n2]

    for atom_name, atom in input_atoms_by_name.items():

        if atom_name not in swap_atom_list:

            output_residue.add(atom)

    for name1, name2 in atom_names_to_swap:

        # rename 1 to 2
        orig_atom = input_atoms_by_name[name1]
        swap_atom = Atom(name2, orig_atom.coord, orig_atom.bfactor,
                         orig_atom.occupancy, orig_atom.altloc, orig_atom.fullname,
                         orig_atom.serial_number, element=orig_atom.element)
        output_residue.add(swap_atom)

        # rename 2 to 1
        orig_atom = input_atoms_by_name[name2]
        swap_atom = Atom(name1, orig_atom.coord, orig_atom.bfactor,
                         orig_atom.occupancy, orig_atom.altloc, orig_atom.fullname,
                         orig_atom.serial_number, element=orig_atom.element)
        output_residue.add(swap_atom)

    return output_residue


def generate_symmetry_alternative(input_residue: Residue) -> Residue:

    if input_residue.resname == "ASP":
        return _dup_residue_but_swap_atoms(input_residue, [("OD1", "OD2")])

    elif input_residue.resname == "GLU":
        return _dup_residue_but_swap_atoms(input_residue, [("OE1", "OE2")])

    elif input_residue.resname == "PHE":
        return _dup_residue_but_swap_atoms(input_residue, [("CD1", "CD2"), ("CE1", "CE2")])

    elif input_residue.resname == "TYR":
        return _dup_residue_but_swap_atoms(input_residue, [("CD1", "CD2"), ("CE1", "CE2")])

    return input_residue


def get_residue_transformations(residues: List[Residue]) -> Tuple[torch.Tensor, torch.Tensor]:

    """
    Returns:
        [n_residues, 3] translations
        [n_residues, 3, 3] rotation matrices
    """

    rotations = []
    translations = []
    for residue in residues:
        atom_n = _get_atom(residue, "N")
        atom_ca = _get_atom(residue, "CA")
        atom_c = _get_atom(residue, "C")

        x1 = torch.tensor(atom_n.coord)
        x2 = torch.tensor(atom_ca.coord)
        x3 = torch.tensor(atom_c.coord)

        v1 = x3 - x2
        v2 = x1 - x2

        e1 = normalize(v1, dim=0)

        u2 = v2 - e1 * torch.matmul(e1.view(3, 1).transpose(1, 0), v2)

        e2 = normalize(u2, dim=0)

        e3 = torch.cross(e1, e2)

        t = x2

        translations.append(t)
        rotations.append(torch.stack([e1, e2, e3]))

    return (torch.stack(translations), torch.stack(rotations))


def get_selected_residues(path: str) -> List[Tuple[str, int, str]]:

    mask = []
    with open(path, 'rt') as file_:
        for line in file_:
            if not line.startswith("#") and len(line.strip()) > 0:

                chain_id, residue_number, three_letter_code = line.split()
                residue_number = int(residue_number)

                mask.append((chain_id, residue_number, three_letter_code))

    return mask


def get_residue_proximities(residues: List[Residue]) -> torch.Tensor:

    residue_count = len(residues)

    residue_proximities = torch.empty((residue_count, residue_count), dtype=float)

    for i in range(residue_count):

        atoms_i = list(residues[i].get_atoms())
        atom_count_i = len(atoms_i)
        atom_positions_i = torch.stack([atom.coord for atom in atoms_i])

        for j in range(residue_count):

            atoms_j = list(residues[j].get_atoms())
            atom_count_j = len(atoms_j)
            atom_positions_j = torch.stack([atom.position for atom in atoms_j])

            atomic_distances_ij = torch.cdist(atom_positions_i, atom_positions_j, p=2)

            min_distance = torch.min(atomic_distances_ij).item()
            proximity = 1.0 / (1.0 + min_distance)

            residue_proximities[i, j] = proximity
            residue_proximities[j, i] = proximity

    return residue_proximities


def get_atom14_positions(residue: Residue) -> Tuple[torch.Tensor, torch.Tensor]:
    atom_names = openfold_residue_atom14_names[residue.get_resname()]

    masks = []
    positions = []
    for atom_name in atom_names:

        if len(atom_name) > 0:
            try:
                atom = _get_atom(residue, atom_name)
                positions.append(atom.coord)
                masks.append(True)

            except Exception as e:
                masks.append(False)
                positions.append((0.0, 0.0, 0.0))

                _log.warning(f"{residue.get_full_id()} not adding 14-formatted position for atom: {str(e)}")
        else:
            masks.append(False)
            positions.append((0.0, 0.0, 0.0))

    return torch.tensor(numpy.array(positions)), torch.tensor(masks)


def recreate_structure(structure_id: str,
                       data_by_chain: List[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]]) -> Structure:
    """
        Args:
            amino_acids_by_chain:
                [0] chain ids
                [1] residue numbers
                [2] amino acids per residue, one-hot encoded
                [3] positions of atoms: [n_residues, n_atoms, 3]
    """

    structure = Structure(structure_id)
    model = Model(0)
    structure.add(model)

    atom_count = 0

    for chain_id, residue_numbers, sequence, atom_positions in data_by_chain:

        chain = Chain(chain_id)
        model.add(chain)

        amino_acids = one_hot_decode_sequence(sequence)

        for residue_index, amino_acid in enumerate(amino_acids):

            if amino_acid is None:
                continue

            residue_name = amino_acid.three_letter_code
            residue_number = residue_numbers[residue_index]

            residue = Residue((" ", residue_number, " "),
                              residue_name, chain_id)

            for atom_index, atom_name in enumerate(openfold_residue_atom14_names[residue_name]):

                if len(atom_name) == 0:
                    continue

                position = atom_positions[residue_index][atom_index]

                atom_count += 1
                atom = Atom(atom_name, position, 0.0, 1.0, " ", atom_name,
                            atom_count, atom_name[0])

                residue.add(atom)

            chain.add(residue)

    return structure


def get_ordered_torsions(residues: List[Residue], index: int) -> Tuple[torch.Tensor, torch.Tensor]:

    residue = residues[index]

    if (index - 1) >= 0:

        prev_residue = residues[index - 1]

        phi = calc_dihedral(Vector(_get_atom(prev_residue, "C").coord),
                            Vector(_get_atom(residue ,"N").coord),
                            Vector(_get_atom(residue, "CA").coord),
                            Vector(_get_atom(residue, "C").coord))
        mask_phi = True
    else:
        phi = 0.0
        mask_phi = False

    if (index + 1) < len(residues):

        next_residue = residues[index + 1]

        omega = calc_dihedral(Vector(_get_atom(residue, "CA").coord),
                              Vector(_get_atom(residue, "C").coord),
                              Vector(_get_atom(next_residue, "N").coord),
                              Vector(_get_atom(next_residue, "CA").coord))
        mask_omega = True

        psi = calc_dihedral(Vector(_get_atom(residue, "N").coord),
                            Vector(_get_atom(residue, "CA").coord),
                            Vector(_get_atom(residue, "C").coord),
                            Vector(_get_atom(next_residue, "N").coord))
        mask_psi = True
    else:
        omega = 0.0
        mask_omega = False
        psi = 0.0
        mask_psi = False

    torsions = [omega, phi, psi]
    mask = [mask_omega, mask_phi, mask_psi]

    amino_acid = amino_acids_by_code[residue.get_resname()]

    for torsion_index, has_torsion in enumerate(openfold_chi_angles_mask[amino_acid.index]):

        mask.append(has_torsion)
        if has_torsion:
            t1, t2, t3, t4 = openfold_chi_angles_atoms[residue.get_resname()][torsion_index]
            torsions.append(calc_dihedral(Vector(_get_atom(residue, t1).coord),
                                          Vector(_get_atom(residue, t2).coord),
                                          Vector(_get_atom(residue, t3).coord),
                                          Vector(_get_atom(residue, t4).coord)))
        else:
            torsions.append(0.0)

    return torch.tensor(torsions), torch.tensor(mask)
