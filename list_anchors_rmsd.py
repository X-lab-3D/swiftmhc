#!/usr/bin/env python

from math import sqrt
from argparse import ArgumentParser
from io import StringIO
import csv

import numpy
import h5py

from PANDORA.PMHC.Anchors import pMHCI_anchors

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Structure import Structure
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom


arg_parser = ArgumentParser()
arg_parser.add_argument("truth_hdf5")
arg_parser.add_argument("pred_hdf5")
arg_parser.add_argument("output_table")

pdb_parser = PDBParser()


def retrieve_pdb(data: h5py.Dataset, id_: str) -> Structure:

    structure_s = "".join([b.decode("utf-8") for b in data.tolist()])

    return pdb_parser.get_structure(id_, StringIO(structure_s))


def get_chain(structure: Structure, chain_id: str) -> Chain:

    for chain in structure.get_chains():
        if chain.id == chain_id:
            return chain

    raise ValueError(f"No such chain: {chain_id}")


def get_atom(residue: Residue, name: str) -> Atom:
    for atom in residue.get_atoms():
        if atom.id == name:
            return atom

    raise ValueError(f"No such atom: {name}")


def get_ca_rmsd(chain1: Chain, chain2: Chain) -> float:

    residues2 = list(chain2.get_residues())

    sum_ = 0.0
    count = 0
    for i, residue1 in enumerate(chain1.get_residues()):
        residue2 = residues2[i]

        ca1 = numpy.array(get_atom(residue1, 'CA').coord)
        ca2 = numpy.array(get_atom(residue2, 'CA').coord)

        sum_ += numpy.square(ca1 - ca2).sum()
        count += 1

    return numpy.sqrt(sum_ / count)


if __name__ == "__main__":

    args = arg_parser.parse_args()

    with open(args.output_table, 'wt') as output_file:
        w = csv.writer(output_file)
        w.writerow(["id", "true anchor 1", "true anchor 2", "predicted anchor 1", "predicted anchor 2", "rmsd"])

    with h5py.File(args.truth_hdf5, 'r') as truth_file:
        ids = list(truth_file.keys())

        with h5py.File(args.pred_hdf5, 'r') as pred_file:

            for id_ in ids:

                if id_ not in pred_file:
                    raise ValueError(f"{id_} missing from {args.pred_hdf5}")

                true_structure = retrieve_pdb(truth_file[id_]["structure"][:], f"true-{id_}")
                pred_structure = retrieve_pdb(pred_file[id_]["structure"][:], f"pred-{id_}")

                rmsd = get_ca_rmsd(get_chain(true_structure, 'P'), get_chain(pred_structure, 'P'))

                true_anchors = pMHCI_anchors(true_structure)
                pred_anchors = pMHCI_anchors(pred_structure)

                if len(true_anchors) != 2:
                    if len(true_anchors) == 1:
                        true_anchors.append(None)
                    else:
                        raise ValueError(f"anchors {true_anchors} for {id_} in {args.truth_hdf5}")

                if len(pred_anchors) != 2:
                    if len(pred_anchors) == 1:
                        pred_anchors.append(None)
                    else:
                        raise ValueError(f"anchors {pred_anchors} for {id_} in {args.pred_hdf5}")

                with open(args.output_table, 'at') as output_file:
                    w = csv.writer(output_file)
                    w.writerow([id_, true_anchors[0], true_anchors[1], pred_anchors[0], pred_anchors[1], round(rmsd, 3)])
