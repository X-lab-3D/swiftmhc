#!/usr/bin/env python

import logging
import os
import sys
from argparse import ArgumentParser
from glob import glob
from tempfile import mkdtemp
import shutil
from math import sqrt

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
arg_parser.add_argument("id", help="id in the name of the structure files n the current working directory, containing changing chain P")
arg_parser.add_argument("template_path", help="template pdb file to use, containing unchanging chain M")


font = ImageFont.truetype("DejaVuSans.ttf", 48)


def get_snapshot_name(id_: str, epoch_index: int, batch_index: int) -> str:
    return f"{id_}-{epoch_index}.{batch_index}.pdb"


def combine_with(m_path: str, p_path: str) -> Structure:
    parser = PDBParser()

    m_struct = parser.get_structure(m_path, m_path)
    p_struct = parser.get_structure(p_path, p_path)

    m_chain = [chain for chain in m_struct.get_chains() if chain.id == "M"][0]
    p_chain = [chain for chain in p_struct.get_chains() if chain.id == "P"][0]

    output_id = os.path.splitext(os.path.basename(p_path))[0]

    output_struct = Structure(output_id)
    output_model = Model("1")
    output_struct.add(output_model)
    output_model.add(m_chain)
    output_model.add(p_chain)

    return output_struct


def find_atom(residue: Residue, name: str) -> Atom:

    for atom in residue.get_atoms():
        if atom.name == name:
            return atom

    raise ValueError(f"not found: {name}")


def get_rmsd(p_path1: str, p_path2: str) -> float:
    parser = PDBParser()

    struct1 = parser.get_structure(p_path1, p_path1)
    struct2 = parser.get_structure(p_path2, p_path2)

    chain1 = [chain for chain in struct1.get_chains() if chain.id == "P"][0]
    chain2 = [chain for chain in struct2.get_chains() if chain.id == "P"][0]

    residues1 = list(chain1.get_residues())
    residues2 = list(chain2.get_residues())

    if len(residues1) != len(residues2):
        raise ValueError("not the same length")

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


def to_frame(structure: Structure, png_path: str, rotation_y: float):

    work_dir = mkdtemp()

    try:
        pdb_path = f"{work_dir}/combined.pdb"

        pdbio = PDBIO()
        pdbio.set_structure(structure)
        pdbio.save(pdb_path)

        pymol_cmd.reinitialize()
        pymol_cmd.load(pdb_path)

        pymol_cmd.hide("cartoon")
        pymol_cmd.show("sphere")

        pymol_cmd.set("transparency", 0.3)
        pymol_cmd.show("surface")

        pymol_cmd.color("blue", "chain M")

        pymol_cmd.rotate("x", -90)

        pymol_cmd.rotate("y", rotation_y)

        pymol_cmd.translate([0, -10, 0])

        pymol_cmd.bg_color("black")
        pymol_cmd.set("opaque_background", "on")
        pymol_cmd.png(png_path, width=1600, height=1200)

    finally:
        shutil.rmtree(work_dir)


def get_snapshot_number(filename: str) -> float:

    name = os.path.splitext(filename)[0]
    epoch_s, batch_s = name.split("-")[-1].split('.')

    return float(epoch_s) + 0.001 * float(batch_s)


if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    args = arg_parser.parse_args()

    png_paths = []

    rotation_y_min = -20.0
    rotation_y_max = 80.0

    snapshot_paths = sorted(glob(f"{args.id}-*.*.pdb"), key=get_snapshot_number)
    if len(snapshot_paths) == 0:
        raise FileNotFoundError(f"no pdbs found matching {args.id}")

    for snapshot_index in range(len(snapshot_paths)):

        frac = float(snapshot_index) / len(snapshot_paths)
        rotation_y = rotation_y_min + (rotation_y_max - rotation_y_min) * frac

        snapshot_path = snapshot_paths[snapshot_index]
        png_path = snapshot_path.replace(".pdb", ".png")

        structure = combine_with(args.template_path, snapshot_path)
        rmsd = get_rmsd(snapshot_path, args.template_path)

        to_frame(structure, png_path, rotation_y)
        add_text(png_path, f"rmsd: {rmsd:.3f}, epoch: {get_snapshot_number(snapshot_path):.3f}")

        png_paths.append(png_path)

        _log.debug(f"created {png_path}")

    clip = ImageSequenceClip(png_paths, 25)
    clip.write_videofile(f"{args.id}.mp4")
