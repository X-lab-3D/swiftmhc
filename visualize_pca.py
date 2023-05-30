#!/usr/bin/env python3

import sys
from typing import Tuple, List
from argparse import ArgumentParser

import numpy
from sklearn.decomposition import PCA
from pdb2sql import pdb2sql
from pymol import cmd, cgo


def get_arrow(position: numpy.ndarray,
              direction: numpy.ndarray,
              color: Tuple[float, float, float]) -> List:

    # arrow parameters
    pos1 = position
    pos2 = position + 25.0 * direction
    pos3 = pos2 + 5.0 * direction
    r1 = 0.5
    r2 = 1.0

    # cgo instructions
    cylinder = [cgo.CYLINDER, pos1[0], pos1[1], pos1[2], pos2[0], pos2[1], pos2[2], r1] + color + color

    cone = [cgo.CONE, pos2[0], pos2[1], pos2[2], pos3[0], pos3[1], pos3[2], r2, 0.0] + color + color + [1.0, 1.0]

    return cylinder + cone


arg_parser = ArgumentParser("visualize pdb PCA")
arg_parser.add_argument("pdb", help="input structure")
arg_parser.add_argument("pse", help="output pymol scene")


if __name__ == "__main__":

    args = arg_parser.parse_args()

    # get atom positions
    pdb = pdb2sql(args.pdb)
    try:

        xyz = pdb.get("x,y,z", name="CA")

    finally:
        pdb._close()

    # calculate pca
    pca = PCA()
    pca.fit(xyz)

    # load structure
    cmd.load(args.pdb)

    # draw the arrows
    arrow1_obj = get_arrow(pca.mean_, pca.components_[0], (1.0, 0.0, 0.0))
    cmd.load_cgo(arrow1_obj, "PC1")

    arrow2_obj = get_arrow(pca.mean_, pca.components_[1], (0.0, 1.0, 0.0))
    cmd.load_cgo(arrow2_obj, "PC2")

    arrow3_obj = get_arrow(pca.mean_, pca.components_[2], (0.0, 0.0, 1.0))
    cmd.load_cgo(arrow3_obj, "PC3")

    # zoom out
    cmd.center("all")
    cmd.zoom("all")

    # save session
    cmd.save(args.pse)
