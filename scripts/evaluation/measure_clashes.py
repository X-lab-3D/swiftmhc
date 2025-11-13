#!/usr/bin/env python

import logging
from typing import List, Union, Tuple
import sys
import os

from argparse import ArgumentParser

arg_parser = ArgumentParser(description="Measure vanderwaals clashes and overlap within pdb files and output a file named clashes.csv. The sum of the vanderwaals Radius overlap and number of clashes will be output to a file named clashes.csv")
arg_parser.add_argument("chain_id", help="ID for the chain to measure clashes on. Clashes will be detected between this chain and all other chains")
arg_parser.add_argument("pdb_files", nargs="+", help="PDB files to measure clashes on. Clashes will be evaluated per PDB file.")

import numpy

from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from Bio.PDB.PDBParser import PDBParser


_log = logging.getLogger(__name__)


class AtomPair:
    def __init__(self, atom1: Atom, atom2: Atom):

        self.atom1 = atom1
        self.atom2 = atom2

    def __eq__(self, other):

        if self.atom1 == other.atom1 and self.atom2 == other.atom2:
            return True

        if self.atom2 == other.atom1 and self.atom1 == other.atom2:
            return True

        return False

    def __hash__(self):

        atom1_str = str(self.atom1.get_parent().get_parent()) + " " + str(self.atom1.get_parent()) + " " + str(self.atom1)
        atom2_str = str(self.atom2.get_parent().get_parent()) + " " + str(self.atom2.get_parent()) + " " + str(self.atom2)

        s = " - ".join(sorted([atom1_str, atom2_str]))

        return hash(s)

    def __iter__(self):

        return iter([self.atom1, self.atom2])


def find_atom(residue: Residue, name: str) -> Union[Atom, None]:

    for atom in residue.get_atoms():
        if atom.get_name() == name:
            return atom

    return None


def get_atoms(residues: List[Residue]) -> List[Atom]:

    atoms = []
    for residue in residues:
        for atom in residue.get_atoms():
            atoms.append(atom)

    return atoms


# Van der Waals radii [Angstroem] of the atoms
van_der_waals_radius = {

    "C": 1.7,
    "N": 1.55,
    "O": 1.52,
    "S": 1.8,
}


def get_radius(atom: Atom) -> float:

    element = atom.element

    return van_der_waals_radius[element]


def get_distance(atom1: Atom, atom2: Atom) -> float:

    p1 = numpy.array(atom1.get_coord())
    p2 = numpy.array(atom2.get_coord())

    return numpy.sqrt(numpy.square(p2 - p1).sum()).item()


def is_backbone(atom: Atom) -> bool:

    return atom.get_name() in {"N", "CA", "C", "O", "OXT"}


def successive_residues(residue1: Residue, residue2: Residue) -> bool:

    if residue1.get_parent() != residue2.get_parent():

        # not on the same chain
        return False

    # hetero flag, number, insertion code
    h1,n1,i1 = residue1.get_id()
    h2,n2,i2 = residue2.get_id()

    icode_seq = ["A", "B", "C", "D", "E", "F", "G"]

    if n1 == n2:

        # same number, check insertion codes

        if len(i1.strip()) > 0 and len(i2.strip()) > 0:

            # both have an insertion code

            # 1A - 1B
            return (abs(icode_seq.index(i1) - icode_seq.index(i2)) == 1)

        elif len(i1.strip()) > 0 and len(i2.strip()) == 0:

            return icode_seq.index(i1) == 0  # 1 - 1A

        elif len(i2.strip()) > 0 and len(i1.strip()) == 0:

            return icode_seq.index(i2) == 0  # 1 - 1A

        else:
            raise ValueError(f"two residues with number {n1} in chain {residue1.get_parent().get_id()} in structure {residue1.get_parent().get_parent().get_id()}")


    elif abs(n1 - n2) == 1:

        return True

    return False


def must_ignore_clash(atom1: Atom, atom2: Atom) -> bool:

    residue1 = atom1.get_parent()
    residue2 = atom2.get_parent()

    element1 = atom1.element
    element2 = atom2.element

    if element1 == "H" or element2 == "H":

        return True

    elif residue1 == residue2:

        return True

    elif residue1.get_resname() == "CYS" and residue2.get_resname() == "CYS":

        # disulfid
        return True

    elif successive_residues(residue1, residue2):

        if is_backbone(atom1) and is_backbone(atom2):

            return True

        elif residue1.get_resname() == "PRO" and is_backbone(atom2) or \
             residue2.get_resname() == "PRO" and is_backbone(atom1):

            return True

    return False


pdb_parser = PDBParser()


def analyse(pdb_path: str, peptide_chain_id: str) -> Tuple[int, float]:

    pdb_name = os.path.basename(pdb_path)

    structure = pdb_parser.get_structure(pdb_name, pdb_path)
    model = structure[0]

    # index atoms
    peptide_atoms = []
    other_atoms = []
    for chain in model:
        if chain.get_id() == peptide_chain_id:
            peptide_atoms = get_atoms(chain.get_residues())
        else:
            other_atoms += get_atoms(chain.get_residues())

    clash_count = 0
    overlap_sum = 0.0
    for i, atom1 in enumerate(peptide_atoms):

        for atom2 in (other_atoms + peptide_atoms[i + 1:]):
            if atom1 == atom2:
                continue

            if must_ignore_clash(atom1, atom2):
                continue

            radius1 = get_radius(atom1)
            radius2 = get_radius(atom2)

            distance = get_distance(atom1, atom2)
            if distance < (radius1 + radius2):

                _log.debug(f"found a clash between {atom1.get_parent()} {atom1} and {atom2.get_parent()} {atom2}")

                clash_count += 1
                overlap_sum += radius1 + radius2 - distance

    return clash_count, overlap_sum


if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    args = arg_parser.parse_args()

    with open("clashes.csv", 'w') as f:
        f.write("model name,number of clashes,sum of vanderwaals radii overlap(Å)\n")

    for pdb_path in args.pdb_files:
        clash_count, overlap_sum = analyse(pdb_path, args.chain_id)

        with open("clashes.csv", 'a') as f:
            f.write(f"{pdb_path},{clash_count},{overlap_sum}\n")
