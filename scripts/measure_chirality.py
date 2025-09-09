#!/usr/bin/env python

import logging
import os
import sys
from argparse import ArgumentParser
from typing import Tuple
import numpy
import pandas
from Bio.PDB.Atom import Atom
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Residue import Residue


arg_parser = ArgumentParser(
    description="Examine chiralities for amino acids within PDB files and output them in a table called chirality.csv."
)
arg_parser.add_argument("pdb_paths", nargs="+", help="list of PDB files to investigate")


_log = logging.getLogger(__name__)

pdb_parser = PDBParser()


def get_atom(residue: Residue, name: str) -> Atom:
    for atom in residue.get_atoms():
        if name == atom.get_name():
            return atom

    raise ValueError("Not found: " + name)


def analyze(
    pdb_path: str,
) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
    """Examine chiralities for one pdb file."""
    id_ = os.path.basename(pdb_path)

    structure = pdb_parser.get_structure(id_, pdb_path)

    residues = []
    for chain in structure[0]:
        residues += list(chain.get_residues())

    res_names = []  # column for residues
    chiralities = []  # column for chirality value

    for residue in residues:
        if residue.get_resname() != "GLY":  # glycines have no chiral center
            # retrieve atom positions:
            n = get_atom(residue, "N").get_coord()
            ca = get_atom(residue, "CA").get_coord()
            c = get_atom(residue, "C").get_coord()

            try:
                cb = get_atom(residue, "CB").get_coord()

            except ValueError as e:
                # C-beta might be missing in structure
                _log.warning(str(e))
                continue

            # Position of C-beta, with respect to C-alpha N and C, determines
            # chirality.
            x = numpy.cross(n - ca, c - ca)
            if numpy.dot(x, cb - ca) > 0.0:
                chiralities.append("L")
            else:
                chiralities.append("D")

            segid, num, icode = residue.get_id()
            resid_ = f"{segid}{num}{icode}".strip()

            res_names.append(
                f"{id_} {residue.get_parent().get_id()} {residue.get_resname()} {resid_}"
            )

    # Convert columns to pandas table.
    return pandas.DataFrame(
        {
            "name": res_names,
            "chirality": chiralities,
        }
    )


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    args = arg_parser.parse_args()

    table = []
    for pdb_path in args.pdb_paths:
        rows = analyze(pdb_path)
        table.append(rows)

    table = pandas.concat(table)

    table.to_csv("chirality.csv", index=False)
