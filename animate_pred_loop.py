#!/usr/bin/env python

import logging
import os
import sys
import re
from typing import Optional
from argparse import ArgumentParser
from glob import glob
from tempfile import mkdtemp
import shutil
from math import sqrt
import h5py
from io import StringIO

from PIL import ImageFont, ImageDraw, Image
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
from Bio.PDB.PDBIO import PDBIO
import pymol.cmd as pymol_cmd


_log = logging.getLogger(__name__)

arg_parser = ArgumentParser(description="combine multiple snapshot structures into an animation")
arg_parser.add_argument("hdf5_path", help="path to the hdf5 data file")
arg_parser.add_argument("--align-ref-pdb", "-r", help="path to a pdb structure to align the video to")


font = ImageFont.truetype("DejaVuSans.ttf", 48)


def is_frame_id(s: str) -> bool:
    return re.match("\d+\.\d+", s) is not None


def find_atom(residue: Residue, name: str) -> Atom:

    for atom in residue.get_atoms():
        if atom.name == name:
            return atom

    raise ValueError(f"not found: {name}")


def get_rmsd(struct1: Structure, struct2: Structure) -> float:

    chain1 = [chain for chain in struct1.get_chains() if chain.id == "P"][0]
    chain2 = [chain for chain in struct2.get_chains() if chain.id == "P"][0]

    residues1 = list(chain1.get_residues())
    residues2 = list(chain2.get_residues())

    if len(residues1) != len(residues2):
        raise ValueError(f"not the same length: {struct1.id} chain P has {len(residues1)} residues, while {struct2.id} chain P has {len(residues2)}")

    sum_ = 0.0
    count = 0

    for residue_index in range(len(residues1)):
        residue1 = residues1[residue_index]
        residue2 = residues2[residue_index]

        for atom1 in residue1.get_atoms():

            if atom1.name == "CA":

                atom2 = find_atom(residue2, atom1.name)

                sum_ += sum((atom1.coord - atom2.coord) ** 2)

                count += 1

    return sqrt(sum_ / count)


def add_text(png_path: str, text: str):

    image = Image.open(png_path)
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), text, (255, 255, 255), font=font)

    image.save(png_path)


def to_frame(structure: Structure, png_path: str, rotation_y: float,
             ref_align_pdb_path: Optional[str] = None):

    work_dir = mkdtemp()

    try:
        pdb_path = f"{work_dir}/combined.pdb"

        pdbio = PDBIO()
        pdbio.set_structure(structure)
        pdbio.save(pdb_path)

        if ref_align_pdb_path:
            pymol_cmd.load(ref_align_pdb_path)
        else:
            pymol_cmd.reinitialize()
        pymol_cmd.load(pdb_path)

        models = pymol_cmd.get_object_list('all')

        if ref_align_pdb_path:
            pymol_cmd.align("combined", "ref")
            pymol_cmd.remove("ref")

        pymol_cmd.create("peptide", "chain P")
        pymol_cmd.create("mhc", "chain M")

        for model in models:
            pymol_cmd.delete(model)

        pymol_cmd.hide("cartoon", "chain P")
        pymol_cmd.show("stick", "chain P")

        pymol_cmd.set("transparency", 0.3)
        pymol_cmd.show("surface", "chain M")

        pymol_cmd.center("chain P")
        pymol_cmd.zoom("all")
        pymol_cmd.color("blue", "chain M")

        pymol_cmd.rotate("y", rotation_y)

        pymol_cmd.translate([0, -10, 0])

        pymol_cmd.bg_color("black")
        pymol_cmd.set("opaque_background", "on")
        pymol_cmd.png(png_path, width=1600, height=1200)

    finally:
        shutil.rmtree(work_dir)


def get_frame_number(frame_id: str) -> float:

    epoch_s, batch_s = frame_id.split('.')

    return float(epoch_s) + 0.001 * float(batch_s)


if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    args = arg_parser.parse_args()

    png_paths = []

    rotation_y_min = -20.0
    rotation_y_max = 20.0

    output_name = os.path.basename(args.hdf5_path).replace(".hdf5", "").replace("-animation", "")
    output_dir = output_name

    if not os.path.isdir(output_name):
        os.mkdir(output_name)

    parser = PDBParser()

    with h5py.File(args.hdf5_path, 'r') as hdf5_file:

        true_structure_s = "".join([b.decode("utf-8") for b in hdf5_file[f"true/structure"][:].tolist()])

        true_structure = parser.get_structure("true", StringIO(true_structure_s))

        frame_ids = sorted(filter(is_frame_id, hdf5_file.keys()), key=get_frame_number)

        for frame_index, frame_id in enumerate(frame_ids):

            frac = float(frame_index) / len(frame_ids)
            rotation_y = rotation_y_min + (rotation_y_max - rotation_y_min) * frac

            png_path = os.path.join(output_name, f"{output_name}-{frame_id}.png")

            snapshot_structure_s = "".join([b.decode("utf-8") for b in hdf5_file[f"{frame_id}/structure"][:].tolist()])

            snapshot_structure = parser.get_structure(frame_id, StringIO(snapshot_structure_s))

            rmsd = get_rmsd(snapshot_structure, true_structure)
            with open(os.path.join(output_name, "rmsd.csv"), 'at') as rmsd_file:
                rmsd_file.write(f"{frame_id},{rmsd:.3f}\n")

            to_frame(snapshot_structure, png_path, rotation_y, args.align_ref_pdb)
            add_text(png_path, f"rmsd: {rmsd:.3f}, epoch: {get_frame_number(frame_id):.3f}")

            png_paths.append(png_path)

            _log.debug(f"created {png_path}")

    clip = ImageSequenceClip(png_paths, 25)
    clip.write_videofile(f"{output_name}.mp4")
