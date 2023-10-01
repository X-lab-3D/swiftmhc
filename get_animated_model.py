#!/usr/bin/env python

from argparse import ArgumentParser
import sys

import h5py


arg_parser = ArgumentParser(description="extract the pdb file from an hdf5 file")
arg_parser.add_argument("hdf5_path", help="hdf5 file containing the data")
arg_parser.add_argument("frame_id", help="name of the frame, containing the pdb file")



if __name__ == "__main__":

    args = arg_parser.parse_args()

    with h5py.File(args.hdf5_path, 'r') as hdf5_file:
        frame_group = hdf5_file[args.frame_id]

        for line in frame_group["structure"][:]:
            line = line.decode("utf_8")
            sys.stdout.write(line)
