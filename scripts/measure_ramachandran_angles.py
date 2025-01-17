#!/usr/bin/env python

from typing import List, Tuple
import logging
from argparse import ArgumentParser
from math import sqrt, sin, cos, pi
import sys
import os

import pandas

from Bio.PDB.PDBParser import PDBParser

from glob import glob


_log = logging.getLogger(__name__)

arg_parser = ArgumentParser(description="measure ramachandran (phi & psi) angles and stores them in the output file: ramachandran.csv")
arg_parser.add_argument("pdb_paths", nargs="+", help="list of pdb files to measure angles on")

pdb_parser = PDBParser()


def analyze(
    pdb_path: str,
) -> Tuple[pandas.DataFrame, pandas.DataFrame]:

    id_ = os.path.basename(pdb_path)

    structure = pdb_parser.get_structure(id_, pdb_path)

    phipsi_pairs = []
    for chain in structure[0]:
        chain.atom_to_internal_coordinates()

        for residue in chain:
            rid = f"{id_} {chain.get_id()} {residue.get_resname()} " + "".join([str(x) for x in residue.get_id()]).strip()

            ric = residue.internal_coord

            phi_angle = ric.get_angle("phi")

            psi_angle = ric.get_angle("psi")

            phipsi_pairs.append((rid, phi_angle, psi_angle))


    names = []
    phis = []
    psis = []
    for rid, phi, psi in phipsi_pairs:
        names.append(rid)
        phis.append(phi)
        psis.append(psi)

    ramachandran_rows = pandas.DataFrame({
        "name": names,
        "ϕ(°)": phis,
        "ψ(°)": psis,
    })

    return (ramachandran_rows)

if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    args = arg_parser.parse_args()

    ramachandran_table = []
    for pdb_path in args.pdb_paths:
        ramachandran_rows = analyze(pdb_path)
        ramachandran_table.append(ramachandran_rows)

    ramachandran_table = pandas.concat(ramachandran_table)

    ramachandran_table.to_csv("ramachandran.csv", index=False)
