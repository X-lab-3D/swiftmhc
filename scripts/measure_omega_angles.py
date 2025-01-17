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
from Bio.PDB.Chain import Chain
from Bio.PDB.Atom import Atom

from glob import glob


_log = logging.getLogger(__name__)

arg_parser = ArgumentParser(description="measure omega angles and stores them in the output file: omegas.csv")
arg_parser.add_argument("pdb_paths", nargs="+", help="list of pdb files to measure angles on")

pdb_parser = PDBParser()


def get_residue_id(residue: Residue) -> str:
    """
    Identify a residue by its resname and number.
    """

    return f"{residue.get_resname()} " + "".join([str(x) for x in residue.get_id()]).strip()


def get_torsion_angle(x: List[List[float]]) -> numpy.ndarray:
    """
    Calculates the torsion angle from 4 atomic positions(x):
    """

    p01 = numpy.array(x[1]) - numpy.array(x[0])
    p12 = numpy.array(x[2]) - numpy.array(x[1])
    p23 = numpy.array(x[3]) - numpy.array(x[2])

    c03 = numpy.cross(p23, -p01)

    # normal of the Newman plane
    p12_norm = numpy.sqrt((p12 ** 2).sum())
    n = p12 / p12_norm

    # Newman projections of the two outer bonds.
    p01_proj = (p01 * n).sum() * n
    p01_plane_proj = p01 - p01_proj
    p01_unit = p01_plane_proj / numpy.sqrt((p01_plane_proj ** 2).sum())

    p23_proj = (p23 * n).sum() * n
    p23_plane_proj = p23 - p23_proj
    p23_unit = p23_plane_proj / numpy.sqrt((p23_plane_proj ** 2).sum())

    # Derive angle from Newman projection.
    direction = (c03 * p12).sum()

    cos_angle = (-p01_unit * p23_unit).sum()
    if numpy.abs(cos_angle) > 1.1:
        raise ValueError(f"invalid cos: {cos_angle}")
    else:
        cos_angle = numpy.clip(cos_angle, -1.0, 1.0)

    angle = 180 * numpy.arccos(cos_angle) / pi
    if direction > 0.0:
        return angle
    else:
        return -angle


def get_atom(residue: Residue, name: str) -> Atom:
    """
    Finds an atom in a residue.
    """

    for atom in residue.get_atoms():
        if name == atom.get_name():
            return atom

    raise ValueError("Not found: " + name)


def analyze(
    pdb_path: str,
) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
    """
    Measure angles in one PDB file.
    """

    id_ = os.path.basename(pdb_path)

    structure = pdb_parser.get_structure(id_, pdb_path)

    # variables to populate the output table:
    chain_ids = []  # chain ID column
    res1_ids = []  # column that identifies the N-terminal residue of a bond
    res2_ids = []  # column that identifies the C-terminal residue of a bond
    omega_angles = []  # column that holds the angles

    for chain in structure[0]:

        # identify chain as PDB name + chain letter
        chain_id = f"{id_} {chain.get_id()}"

        prev_residue = None
        for residue in chain:

            # identify residue
            residue_id = get_residue_id(residue)

            # the residue's atoms:
            ca = get_atom(residue, 'CA').get_coord()
            n = get_atom(residue, 'N').get_coord()
            c = get_atom(residue, 'C').get_coord()

            if prev_residue is not None:
                # not the N-terminal residue, so there's a peptide bond

                prev_residue_id = get_residue_id(prev_residue)

                # The atoms of the residue located on the N-terminal side of the bond
                ca_prev = get_atom(prev_residue,'CA').get_coord()
                c_prev = get_atom(prev_residue,'C').get_coord()
                n_prev = get_atom(prev_residue,'N').get_coord()

                # populate table
                chain_ids.append(chain_id)
                res1_ids.append(prev_residue_id)
                res2_ids.append(residue_id)
                omega_angles.append(get_torsion_angle([ca_prev, c_prev, n, ca]))

            prev_residue = residue

    # return as pandas table
    return pandas.DataFrame({
        "chain": chain_ids,
        "residue 1": res1_ids,
        "residue 2": res2_ids,
        "ω(°)": omega_angles,
    })

if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    args = arg_parser.parse_args()

    table = []
    for pdb_path in args.pdb_paths:
        rows = analyze(pdb_path)
        table.append(rows)

    table = pandas.concat(table)

    table.to_csv("omegas.csv", index=False)

