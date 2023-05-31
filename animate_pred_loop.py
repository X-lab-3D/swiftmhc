#!/usr/bin/env python

import os
from argparse import ArgumentParser
from glob import glob
from tempfile import mkdtemp
import shutil

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.PDBIO import PDBIO
import pymol.cmd as pymol_cmd

arg_parser = ArgumentParser(description="combine multiple snapshot structures into an animation")
arg_parser.add_argument("id", help="id in the name of the structure files")
arg_parser.add_argument("template_path", help="template pdb file to use")


def get_snapshot_name(id_: str, epoch_index: int, batch_index: int) -> str:
    return f"{id_}-{epoch_index}.{batch_index}.pdb"


def combine_with(m_path: str, p_path: str) -> Structure:
    parser = PDBParser()

    m_struct = parser.get_structure(m_path, m_path)
    p_struct = parser.get_structure(m_path, p_path)

    m_chain = [chain for chain in m_struct.get_chains() if chain.id == "M"][0]
    p_chain = [chain for chain in p_struct.get_chains() if chain.id == "P"][0]

    output_id = os.path.splitext(os.path.basename(p_path))[0]

    output_struct = Structure(output_id)
    output_model = Model("1")
    output_struct.add(output_model)
    output_model.add(m_chain)
    output_model.add(p_chain)

    return output_struct


def to_frame(structure: Structure, id_: str, epoch_index: int) -> str:

    work_dir = mkdtemp()

    try:
        pdb_path = f"combined-{id_}-e{epoch_index}.pdb"
        png_path = f"{id_}-e{epoch_index}.png"

        pdbio = PDBIO()
        pdbio.set_structure(structure)
        pdbio.save(pdb_path)

        pymol_cmd.reinitialize()
        pymol_cmd.load(pdb_path)

        pymol_cmd.hide("cartoon", "chain P")
        pymol_cmd.show("stick", "chain P")
        pymol_cmd.show("stick", "name CA or sidechain")

        pymol_cmd.color("blue", "chain M")

        pymol_cmd.rotate("x", -90)

        pymol_cmd.translate([0, -10, 0])

        pymol_cmd.png(png_path, width=1600, height=1200)

        return png_path

    finally:
        shutil.rmtree(work_dir)


if __name__ == "__main__":

    args = arg_parser.parse_args()

    png_paths = []

    epoch_index = 0
    batch_index = 0

    while os.path.isfile(get_snapshot_name(args.id, epoch_index, batch_index)):
        while os.path.isfile(get_snapshot_name(args.id, epoch_index, batch_index)):
            snapshot_path = get_snapshot_name(args.id, epoch_index, batch_index)

            structure = combine_with(args.template_path, snapshot_path)

            png_paths.append(to_frame(structure, args.id, epoch_index))

            batch_index += 1

        batch_index = 0
        epoch_index += 1

    clip = ImageSequenceClip(png_paths, 10)
    clip.write_videofile(f"{args.id}.mp4")
