#!/usr/bin/env python

import sys
import logging
from argparse import ArgumentParser
from datetime import datetime
from glob import glob
import gzip
from typing import Set
import os

import pandas


_log = logging.getLogger(__name__)


arg_parser = ArgumentParser(description="counts how many X-ray structures are shared between train (AlphaFold2, AlphaFold2-FineTune, MHCfold) and test sets." +
                                        " This script generates output files in the current directory.")
arg_parser.add_argument("mmcif_directory", help="directory where the entire PDB is stored in mmCIF format")
arg_parser.add_argument("alphafold_finetune_templates_directory", help="directory where AlphaFold2-FineTune templates are stored")
arg_parser.add_argument("test_xray_table", help="CSV file where the test set X-ray structures are listed under the column 'PDBID'. This script will examine the overlap between the train sets and this test set.")


def get_ft2_templates(alphafold_templates_dir: str) -> Set[str]:

    pdbids = set([])

    for filename in os.listdir(alphafold_templates_dir):

        name, ext = os.path.splitext(filename.lower())

        if ext == ".pdb":

            if '_' in name:

                pdbid = name.split('_')[0]

                pdbids.add(pdbid)
            else:
                pdbids.add(name)

    return pdbids


def get_test_xrays(test_xray_table: str) -> Set[str]:

    table = pandas.read_csv(test_xray_table)

    return {pdbid.lower() for pdbid in table['PDBID']}


def get_alphafold2_mhcfold_train_xrays(mmcif_directory: str) -> Set[str]:
    """
    Check every PDB entry (mmCIF file) to see if it meets the criteria.
    """

    af2_date = datetime.strptime("2018-04-30", "%Y-%m-%d")
    mhcfold_date = datetime.strptime("2021-11-1", "%Y-%m-%d")

    af2_pdbids = set([])
    mhcfold_pdbids = set([])
    for path in glob(os.path.join(mmcif_directory, "????.cif.gz")):

        pdbid = os.path.basename(path).split('.')[0].lower()

        r = None
        with gzip.open(path, 'rt') as mmcif_file:

            for line in mmcif_file:

                if line.startswith("_reflns.d_resolution_high "):

                    ss = line.split()
                    if len(ss) != 2:
                        continue

                    if ss[1].strip() != "?" and ss[1].strip() != "." and ss[1].strip() != "":
                        r = float(ss[1])

                    break

        with gzip.open(path, 'rt') as mmcif_file:

            for line in mmcif_file:

                if line.startswith("_pdbx_database_status.recvd_initial_deposition_date "):
                    key, date_s = line.split()

                    date = datetime.strptime(date_s, "%Y-%m-%d")

                    if date < af2_date:
                        af2_pdbids.add(pdbid)

                    if date < mhcfold_date and r is not None and r <= 3.5:
                        mhcfold_pdbids.add(pdbid)

                    break

    return af2_pdbids, mhcfold_pdbids


if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    args = arg_parser.parse_args()

    open("test-xray-pdbids.txt", 'w').write("\n".join(get_test_xrays(args.test_xray_table)))
    _log.info("written test-xray-pdbids.txt")

    open("train-ft2-pdbids.txt", 'w').write("\n".join(get_ft2_templates(args.alphafold_finetune_templates_directory)))
    _log.info("written train-ft2-pdbids.txt")

    af2_pdbids, mhcfold_pdbids = get_alphafold2_mhcfold_train_xrays(args.mmcif_directory)

    open("train-af2-pdbids.txt", 'w').write("\n".join(af2_pdbids))
    _log.info("written train-af2-pdbids.txt")

    open("train-mhcfold-pdbids.txt", 'w').write("\n".join(mhcfold_pdbids))
    _log.info("written train-mhcfold-pdbids.txt")
