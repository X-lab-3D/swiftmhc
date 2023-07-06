#!/usr/bin/env python

import logging
import os
import sys
from argparse import ArgumentParser
from glob import glob
from tempfile import mkdtemp
import shutil
from math import sqrt
from io import StringIO

from Bio.PDB.PDBParser import PDBParser
import pymol.cmd as pymol_cmd
import h5py


_log = logging.getLogger(__name__)

arg_parser = ArgumentParser(description="combine multiple snapshot structures into an animation")
arg_parser.add_argument("hdf5_path", help="path to the hdf5 data file")


def get_frame_number(frame_id: str) -> float:

    epoch_s, batch_s = frame_id.split('.')

    return float(epoch_s) + 0.001 * float(batch_s)


if __name__ == "__main__":

    pdb_parser = PDBParser()

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    args = arg_parser.parse_args()

    png_paths = []

    rotation_y_min = -20.0
    rotation_y_max = 20.0

    output_name = os.path.basename(args.hdf5_path).replace(".hdf5", "")

    with h5py.File(args.hdf5_path, 'r') as hdf5_file:

        frame_ids = sorted(hdf5_file.keys(), key=get_frame_number)
        last_frame_id = frame_ids[-1]

        pdb_path = f"{output_name}-{last_frame_id}.pdb"

        snapshot_structure_s = "".join([b.decode("utf-8") for b in hdf5_file[f"{last_frame_id}/structure"][:].tolist()])

        with open(pdb_path, 'wt') as pdb_file:
            pdb_file.write(snapshot_structure_s)

        print("written:", pdb_path)

        protein_chain_id = "M"
        loop_chain_id = "P"

        cross_attention = hdf5_file[f"{last_frame_id}/cross_attention"][0, 0, ...]

        cross_attention_protein = cross_attention.sum(axis=0)
        cross_attention_protein = cross_attention_protein / cross_attention_protein.max()

        protein_residue_numbers = hdf5_file[f"{last_frame_id}/protein_residue_numbers"][:]
        protein_residue_mask = hdf5_file[f"{last_frame_id}/protein_cross_residue_mask"][:]

    pymol_cmd.reinitialize()
    pymol_cmd.load(pdb_path)

    pymol_cmd.show("sticks", "name CA or sidechain")
    pymol_cmd.show("sticks", f"chain {loop_chain_id}")
    pymol_cmd.hide("cartoon", f"chain {loop_chain_id}")

    pymol_cmd.color("blue", f"chain {protein_chain_id}")

    for residue_index, residue_number in enumerate(protein_residue_numbers):

        if protein_residue_mask[residue_index]:

            attention_value = cross_attention_protein[residue_index]

            rgb = (attention_value, attention_value, 1.0)

            color_name = f"protein-residue-{residue_index}"
            pymol_cmd.set_color(color_name, rgb)
            pymol_cmd.color(color_name, f"chain {protein_chain_id} and residue {residue_number}")

    pymol_cmd.color("green", f"chain {loop_chain_id}")

    session_path = f"{output_name}-{last_frame_id}.pse"
    pymol_cmd.save(session_path)
    print("written:", session_path)

