#!/usr/bin/env python

from argparse import ArgumentParser
from glob import glob
from typing import Set, Tuple
from datetime import datetime
import pandas
import os

from PANDORA import Database

from Bio.PDB.PDBParser import PDBParser


arg_parser = ArgumentParser(description="counts how many X-ray structures are shared between train (AlphaFold2, AlphaFold2-FineTune, MHCfold) and test sets." +
                                        " This script generates output files in the current directory:\n" +
                                        "  - test_peptides.txt\n" +
                                        "  - test_peptides_overlapping_with_alphafold2_training.txt\n" +
                                        "  - test_peptides_overlapping_with_mhcfold_training.txt\n" +
                                        "  - test_peptides_overlapping_with_alphafold2_finetune_training.txt\n")
arg_parser.add_argument("pandora_database", help="path to PANDORA_database.pkl file")
arg_parser.add_argument("pdb_directory", help="directory where the PDB files are stored of the entire PDB")
arg_parser.add_argument("alphafold_finetune_templates_directory", help="directory where AlphaFold2-FineTune templates are stored")
arg_parser.add_argument("test_xray_table", help="CSV file where the test set X-ray structures are listed under the column 'PDBID'. This script will examine the overlap between the train sets and this test set.")


mhcfold_date = datetime.strptime("2021-11-1", "%Y-%m-%d")
mhcfold_resolution = 3.5

af2_date = datetime.strptime("2018-04-30", "%Y-%m-%d")

# PDB entries that moved to a different ID
swap_table = {
    "3GJG": "3HAE",
    "1JTR": "1MWA",
    "2YF5": "4DOC",
    "2YF6": "4DOB",
    "3NFJ": "3NFN",
    "4ZEZ": "5JZI",
    "5C0H": "5N1Y",
    "5CNZ": "5H5Z",
}

# not in RCSB anymore
removed = {"1L6Q", "4EUQ"}

pdb_parser = PDBParser(QUIET=True)


def get_ft2_templates(directory_path: str) -> Set[str]:

    pdbids = set([])

    for filename in os.listdir(directory_path):

        name, ext = os.path.splitext(filename.lower())

        if ext == ".pdb":

            if '_' in name:

                pdbid = name.split('_')[0]

                pdbids.add(pdbid.upper())
            else:
                pdbids.add(name)

    return pdbids


def peptides_to_file(path: str, peptides: Set[str]):
    with open(path, 'wt') as f:
        for peptide in peptides:
            f.write(peptide + "\n")


def pmhcs_to_file(path: str, pmhcs: Set[Tuple[str, str]]):
    with open(path, 'wt') as f:
        for peptide, mhc in pmhcs:
            f.write(f"{peptide} {mhc}\n")

if __name__ == "__main__":

    args = arg_parser.parse_args()

    pandora_database = Database.load(args.pandora_database)

    table = pandas.read_csv(args.test_xray_table)

    ft2_pmhcs = set([])
    af2_pmhcs = set([])
    af2_ft2_pmhcs = set([])

    ft2_pdbids = get_ft2_templates(args.alphafold_finetune_templates_directory)
    for pdbid, template in pandora_database.MHCI_data.items():

        if pdbid in removed:
            continue

        pdbid = swap_table.get(pdbid, pdbid)

        path = os.path.join(args.pdb_directory, f"pdb{pdbid.lower()}.ent")
        pdb = pdb_parser.get_structure(pdbid, path)

        date = datetime.strptime(pdb.header["deposition_date"], "%Y-%m-%d")

        if date < af2_date:

            af2_pmhcs.add((template.peptide, template.M_chain_seq))

            if pdbid in ft2_pdbids:
                af2_ft2_pmhcs.add((template.peptide, template.M_chain_seq))

        if pdbid in ft2_pdbids:

            ft2_pmhcs.add((template.peptide, template.M_chain_seq))

    test_peptides_overlapping_af2 = set([])
    test_peptides_overlapping_ft = set([])

    test_peptides_overlapping_mhcfold = set([])
    test_peptides = set([])
    for _, row in table.iterrows():

        pdbid = row.PDBID
        peptide = row.peptide

        pdbid = swap_table.get(pdbid, pdbid)

        path = os.path.join(args.pdb_directory, f"pdb{pdbid.lower()}.ent")
        pdb = pdb_parser.get_structure(pdbid, path)

        date = datetime.strptime(pdb.header["deposition_date"], "%Y-%m-%d")
        resolution = float(pdb.header["resolution"])

        if date < mhcfold_date and resolution < mhcfold_resolution:

            test_peptides_overlapping_mhcfold.add(peptide)

        if date < af2_date :

            test_peptides_overlapping_af2.add(peptide)

        if pdbid in ft2_pdbids:

            test_peptides_overlapping_ft.add(peptide)

        test_peptides.add(peptide)

    peptides_to_file("test_peptides.txt", test_peptides)
    peptides_to_file("test_peptides_overlapping_with_alphafold2_training.txt", test_peptides_overlapping_af2)
    peptides_to_file("test_peptides_overlapping_with_mhcfold_training.txt", test_peptides_overlapping_mhcfold)
    peptides_to_file("test_peptides_overlapping_with_alphafold2_finetune_training.txt", test_peptides_overlapping_ft)

    print("unique AF2 training pMHC-I's", len(af2_pmhcs))
    print("unique AF2-FT training pMHC-I's", len(ft2_pmhcs))
    pmhcs_to_file("unique_alphafold2_pmhci.txt", af2_pmhcs)
    pmhcs_to_file("unique_alphafold2_finetune_pmhci.txt", ft2_pmhcs)
    print("unique AF2 training pMHC-I's overlapping with AF2-FT training pMHC-I's", len(af2_ft2_pmhcs))
    pmhcs_to_file("unique_alphafold2_alphafold2_finetune_pmhci.txt", af2_ft2_pmhcs)

    print("unique test peptides overlapping with AF2 training:", len(test_peptides_overlapping_af2))
    print("unique test peptides not overlapping with AF2 training:", len(test_peptides - test_peptides_overlapping_af2))

    print("unique test peptides overlapping with AF2FT training:", len(test_peptides_overlapping_ft))
    print("unique test peptides not overlapping with AF2FT training:", len(test_peptides - test_peptides_overlapping_ft))

    print("unique test peptides overlapping with AF2FT training or AF2 training:", len(test_peptides_overlapping_af2 | test_peptides_overlapping_ft))
    print("unique test peptides overlapping with AF2FT training and AF2 training:", len(test_peptides_overlapping_af2 & test_peptides_overlapping_ft))
    print("unique test peptides not overlapping with AF2FT training or AF2 training:", len(test_peptides - test_peptides_overlapping_af2 - test_peptides_overlapping_ft))

    print("unique peptides overlapping with MHCfold training:",len(test_peptides_overlapping_mhcfold))
    print("unique peptides not overlapping with MHCfold training:",len(test_peptides - test_peptides_overlapping_mhcfold))

