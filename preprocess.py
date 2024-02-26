#!/usr/bin/env python3

import shutil
from typing import List
from tempfile import mkdtemp
from multiprocessing import Process
import logging
from argparse import ArgumentParser
import os

from uuid import uuid4

import pandas
import h5py
from swiftmhc.preprocess import preprocess


_log = logging.getLogger(__name__)


arg_parser = ArgumentParser(description="preprocess data for tcrspec to operate on")
arg_parser.add_argument("table_path", help="table with input data, containing: ID, allele (optional), measurement_value (optional)")
arg_parser.add_argument("reference_structure", help="reference structure, to which the masks apply")
arg_parser.add_argument("models_dir", help="where the input models are stored: pdb files containing chain M (for protein) and P (for peptide)")
arg_parser.add_argument("protein_self_mask", help="TSV file with mask data, to indicate which protein residues to use for self attention, row format: CHAIN_ID  RESIDUE_NUMBER  AMINO_ACID_THREE_LETTER_CODE")
arg_parser.add_argument("protein_cross_mask", help="TSV file with mask data, to indicate which protein residues to use for cross attention, row format: CHAIN_ID  RESIDUE_NUMBER  AMINO_ACID_THREE_LETTER_CODE")
arg_parser.add_argument("output_path", help="hdf5 file where to store the preprocessed data")
arg_parser.add_argument("--debug", "-d", help="adds debug statments to the log", action="store_const", const=True, default=False)
arg_parser.add_argument("--processes", "-p", help="number of simultaneous processes", type=int, default=1)


def split_table(table_path: str, dir_path: str, split_count: int) -> List[str]:
    """
    Splits one table into multiple tables.

    Args:
        table_path: input table, to be split
        dir_path: directory where to store the split tables
        split_count: how many tables to split into

    Returns: the list of created tables
    """

    table_name = os.path.splitext(os.path.basename(table_path))[0]
    table = pandas.read_csv(table_path)
    table_size = table.shape[0]
    split_size = int(table_size / split_count)

    paths = []
    offset = 0
    for split_index in range(split_count):

        path = os.path.join(dir_path, f"{table_name}_{split_index}.csv")
        paths.append(path)

        if (split_index + 1) == split_count:  # last?

            table[offset: ].to_csv(path, index=False)
        else:
            table[offset: offset + split_size].to_csv(path, index=False)
            offset += split_size

    return paths


if __name__ == "__main__":

    args = arg_parser.parse_args()

    logging.basicConfig(filename="preprocess.log", filemode="a", level=logging.DEBUG if args.debug else logging.INFO)

    if args.processes == 1:
        preprocess(
            args.table_path,
            args.models_dir,
            args.protein_self_mask,
            args.protein_cross_mask,
            args.output_path,
            args.reference_structure,
        )
    else:
        tmp_dir = os.path.join(os.path.dirname(args.output_path), uuid4().hex)
        os.mkdir(tmp_dir)

        try:
            # split the data
            table_paths = split_table(args.table_path, tmp_dir, args.processes)

            ps = []
            tmp_output_paths = []
            for table_path in table_paths:

                output_path = f"{table_path}.hdf5"

                p = Process(target=preprocess, args=(table_path, args.models_dir, args.protein_self_mask, args.protein_cross_mask, output_path, args.reference_structure))
                ps.append(p)

                tmp_output_paths.append(output_path)
                p.start()

            # wait for all processes to complete
            for p in ps:
                p.join()

            # merge the preprocessed data into one file
            with h5py.File(args.output_path, 'w') as output_file:
                for tmp_output_path in tmp_output_paths:
                    if os.path.isfile(tmp_output_path):
                        with h5py.File(tmp_output_path, 'r') as tmp_file:
                            for key, value in tmp_file.items():
                                if key not in output_file:
                                    tmp_file.copy(value, output_file)

                        os.remove(tmp_output_path)
        finally:
            shutil.rmtree(tmp_dir)
