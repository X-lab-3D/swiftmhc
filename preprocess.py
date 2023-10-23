#!/usr/bin/env python3

import logging
from argparse import ArgumentParser

from tcrspec.preprocess import preprocess


_log = logging.getLogger(__name__)


arg_parser = ArgumentParser(description="preprocess data for tcrspec to operate on")
arg_parser.add_argument("table_path", help="table with input data")
arg_parser.add_argument("models_dir", help="where the models are stored")
arg_parser.add_argument("protein_self_mask", help="file with mask data, to indicate which protein residues to use for self attention")
arg_parser.add_argument("protein_cross_mask", help="file with mask data, to indicate which protein residues to use for cross attention")
arg_parser.add_argument("output_path", help="hdf5 file where to store the data")
arg_parser.add_argument("--debug", "-d", help="adds debug statments", action="store_const", const=True, default=False)

if __name__ == "__main__":

    args = arg_parser.parse_args()

    logging.basicConfig(filename="preprocess.log", filemode="a", level=logging.DEBUG if args.debug else logging.INFO)

    preprocess(args.table_path,
               args.models_dir,
               args.protein_self_mask,
               args.protein_cross_mask,
               args.output_path)
