#!/usr/bin/env python

import sys
import os
from typing import List, Dict
from glob import glob
from argparse import ArgumentParser

import pandas
import numpy
import pymol.cmd as pymol_cmd

from tcrspec.domain.amino_acid import amino_acids_by_name


arg_parser = ArgumentParser(description="color a yasara scene from weight values")
arg_parser.add_argument("weight_table", help="CSV table with weight values")
arg_parser.add_argument("input_scene", help="the scene to modify")
arg_parser.add_argument("epitope_residue", type=int, help="the epitope residue to use the weights from")


def get_values(data_frame: pandas.DataFrame, epitope_residue_number: int):

    data = data_frame.values
    values = data[:, epitope_residue_number - 1].astype(numpy.float64)

    min_value = numpy.min(values)
    max_value = numpy.max(values)

    values = (values - min_value) / (max_value - min_value)

    values_per_residue = {data_frame.index[i]: values[i].item() for i in range(len(data_frame.index))}
    return values_per_residue


def make_scene(csv_path: str, input_session_path: str, epitope_residue_number: int):

    file_prefix = os.path.splitext(csv_path)[0]

    data_frame = pandas.read_csv(csv_path, sep=',', header=0, index_col=0)

    values = get_values(data_frame, epitope_residue_number)

    pymol_cmd.load(input_session_path)
    pymol_cmd.color("blue")
    pymol_cmd.color("green", f"chain C and res {epitope_residue_number}")

    for residue_id, value in values.items():
        if "<gap>" not in residue_id:
            residue_number, amino_acid_name = residue_id.split()

            amino_acid = amino_acids_by_name[amino_acid_name]

            r = 1.0
            g = 1.0 - value
            b = 1.0 - value

            pymol_cmd.set_color(f"color_for_{residue_number}", f"[{r:.1f}, {g:.1f}, {b:.1f}]")
            pymol_cmd.color(f"color_for_{residue_number}", f"chain A and res {residue_number}")

    session_path = f"{file_prefix}-epitope-{epitope_residue_number}.pse"
    image_path = f"{file_prefix}-epitope-{epitope_residue_number}.png"

    pymol_cmd.save(session_path)
    pymol_cmd.png(image_path)

    print("written:", [session_path, image_path])


if  __name__ == "__main__":

    args = arg_parser.parse_args()

    make_scene(args.weight_table, args.input_scene, args.epitope_residue)
