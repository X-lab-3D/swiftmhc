#!/usr/bin/env python

import sys
import os
from typing import List, Dict
from glob import glob
from argparse import ArgumentParser

import pandas
import numpy
import yasara

from tcrspec.domain.amino_acid import amino_acids_by_name


arg_parser = ArgumentParser(description="color a yasara scene from weight values")
arg_parser.add_argument("weight_table", help="CSV table with weight values")
arg_parser.add_argument("input_scene", help="the scene to modify")
arg_parser.add_argument("epitope_residue", type=int, help="the epitope residue to use the weights from")


def get_hues(data_frame: pandas.DataFrame, epitope_residue_number: int):

    data = data_frame.values
    values = data[:, epitope_residue_number - 1]

    min_value = numpy.min(values)
    max_value = numpy.max(values)

    hues = 120 + 120 * (values - min_value) / (max_value - min_value)

    hues_per_residue = {data_frame.index[i]: int(hues[i].item()) for i in range(len(data_frame.index))}
    return hues_per_residue


def make_scene(csv_path: str, input_scene_path: str, epitope_residue_number: int):

    file_prefix = os.path.splitext(csv_path)[0]

    data_frame = pandas.read_csv(csv_path, sep=',', header=0, index_col=0)

    hues = get_hues(data_frame, epitope_residue_number)

    yasara.Clear()
    yasara.LoadSce(input_scene_path)
    yasara.ColorObj(1, "white")
    for residue_id, hue in hues.items():
        if "<gap>" not in residue_id:
            residue_number, amino_acid_name = residue_id.split()

            amino_acid = amino_acids_by_name[amino_acid_name]

            yasara.ColorRes(f"{amino_acid.three_letter_code} {residue_number} and mol A", hue)

    scene_path = f"{file_prefix}-epitope-{epitope_residue_number}.sce"
    image_path = f"{file_prefix}-epitope-{epitope_residue_number}.png"

    yasara.SaveSce(scene_path)
    yasara.RayTrace(image_path)

    print("written:", [scene_path, image_path])


if  __name__ == "__main__":

    args = arg_parser.parse_args()

    make_scene(args.weight_table, args.input_scene, args.epitope_residue)
