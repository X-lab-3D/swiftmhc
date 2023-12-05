#!/usr/bin/env python

from argparse import ArgumentParser
import csv
from typing import List

import h5py

from tcrspec.domain.amino_acid import amino_acids_by_one_hot_index


arg_parser = ArgumentParser(description="list the sequences in a preprocessed hdf5 file")
arg_parser.add_argument("preprocessed_file")
arg_parser.add_argument("output_table")


def get_sequence(indices: List[int]) -> str:
    return "".join([amino_acids_by_one_hot_index[i].one_letter_code
                    for i in indices])


if __name__ == "__main__":

    args = arg_parser.parse_args()

    list_path = args.output_table
    with open(list_path, 'wt') as f:
        w = csv.writer(f)
        w.writerow(["ID", "loop", "protein"])

    preprocessed_path = args.preprocessed_file
    with h5py.File(preprocessed_path, 'r') as f5:
        for id_ in f5.keys():
            loop_aatype = f5[id_]["loop/aatype"][:]
            loop_sequence = get_sequence(loop_aatype)

            protein_aatype = f5[id_]["protein/aatype"][:]
            protein_sequence = get_sequence(protein_aatype)

            with open(list_path, 'at') as f:
                w = csv.writer(f)
                w.writerow([id_, loop_sequence, protein_sequence])

