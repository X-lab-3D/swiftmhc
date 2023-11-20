#!/usr/bin/env python

from argparse import ArgumentParser

import h5py


arg_parser = ArgumentParser()
arg_parser.add_argument("hdf5")
arg_parser.add_argument("structure")


if __name__ == "__main__":

    args = arg_parser.parse_args()

    with h5py.File(args.hdf5, 'r') as hdf5:
        data = hdf5[args.structure]["structure"][:]

        for line in data:
            print(line.decode("utf_8").strip())
