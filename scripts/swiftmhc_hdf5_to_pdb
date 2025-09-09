#!/usr/bin/env python

import os
from argparse import ArgumentParser
import h5py


arg_parser = ArgumentParser(
    description="SwiftMHC outputs structures in an HDF5 file. This script converts that HDF5 file to PDB files"
)
arg_parser.add_argument("hdf5_path", help="input HDF5 file")


if __name__ == "__main__":
    args = arg_parser.parse_args()

    dir_path = args.hdf5_path.replace(".hdf5", "")
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    count = 0
    with h5py.File(args.hdf5_path, "r") as hdf5_file:
        for pdbid in hdf5_file:
            path = os.path.join(dir_path, f"{pdbid}.pdb")

            structure = ""
            for arr in hdf5_file[pdbid]["structure"][:]:
                structure += bytes(arr).decode("utf_8").strip() + "\n"

            with open(path, "wt") as output_file:
                output_file.write(structure)

            count += 1

    print(f"output {count} PDB files to {dir_path}")
