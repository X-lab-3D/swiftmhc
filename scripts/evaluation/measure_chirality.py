#!/usr/bin/env python

from typing import List, Tuple
import logging
from argparse import ArgumentParser
from math import sqrt, sin, cos, pi
import sys
import os

import numpy
import pandas

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom

from glob import glob


_log = logging.getLogger(__name__)

pdb_parser = PDBParser()

arg_parser = ArgumentParser(description="measure chiralities on a series of pdb files, outputs to a file names chirality.csv")
arg_parser.add_argument("pdb_paths", nargs="+", help="input pdb files to measure chiralities on")


def mean(values: List[float]) -> float:
    return sum(values) / len(values)


def square(x: float) -> float:
    return x * x


def get_atom(residue: Residue, name: str) -> Atom:

    for atom in residue.get_atoms():
        if name == atom.get_name():
            return atom

    raise ValueError("Not found: " + name)


def analyze(
    pdb_path: str,
) -> Tuple[pandas.DataFrame, pandas.DataFrame]:

    id_ = os.path.basename(pdb_path)

    structure = pdb_parser.get_structure(id_, pdb_path)

    if "P" in structure[0]:
        mhc_chain = structure[0]["M"]
        peptide_chain = structure[0]["P"]
        residues = list(peptide_chain.get_residues()) + list(mhc_chain.get_residues())

    elif "C" in structure[0]:
        mhc_chain = structure[0]["A"]
        peptide_chain = structure[0]["C"]
        residues = list(peptide_chain.get_residues()) + list(mhc_chain.get_residues())

    elif "B" in structure[0]:
        mhc_chain = structure[0]["A"]
        peptide_chain = structure[0]["B"]
        residues = list(peptide_chain.get_residues()) + list(mhc_chain.get_residues())

    else:
        residues = []
        for chain in structure[0]:
            residues += list(chain.get_residues())

    names = []
    chiralities = []
    for residue in residues:

        if residue.get_resname() != "GLY":

            try:
                n = get_atom(residue, 'N').get_coord()
                ca = get_atom(residue, 'CA').get_coord()
                c = get_atom(residue, 'C').get_coord()

                cb = get_atom(residue, 'CB').get_coord()

            except ValueError as e:
                _log.warning(str(e))
                continue

            x = numpy.cross(n - ca, c - ca)
            if numpy.dot(x, cb - ca) > 0.0:
                chiralities.append("L")
            elif numpy.dot(x, cb - ca) < 0.0:
                chiralities.append("D")
            else:
                chiralities.append("?")

            segid, num, icode = residue.get_id()
            resid_ = f"{segid}{num}{icode}".strip()

            names.append(f"{id_} {residue.get_parent().get_id()} {residue.get_resname()} {resid_}")

    return pandas.DataFrame({
        "name": names,
        "chirality": chiralities,
    })

if __name__ == "__main__":

    logging.basicConfig(filename="chirality.log", level=logging.INFO)

    args = arg_parser.parse_args()

    count_undetermined = 0
    table = []
    for pdb_path in args.pdb_paths:
        rows = analyze(pdb_path)
        table.append(rows)

    table = pandas.concat(table)

    table.to_csv("chirality.csv", index=False)

