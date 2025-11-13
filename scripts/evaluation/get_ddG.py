#!/usr/bin/env python

import logging
from argparse import ArgumentParser
from math import log, isnan
import sys

import numpy
import pandas


_log = logging.getLogger(__name__)


arg_parser = ArgumentParser(description="Calculates ΔΔG values for a series of mutants and outputs them to a table")
arg_parser.add_argument("mutants_table", help="CSV file that holds the data on the mutants to be investigated. This is output by the 'find_mutations.py' script. It needs to have the columns: mut_ID (an identifier), wt_ID (an identifier), allele, wt_peptide, mut_peptide, wt_value (true Kd or IC50), mut_value (true Kd or IC50)")
arg_parser.add_argument("predictions_table", nargs="+", help="CSV file output by SwiftMHC, containing the predicted BA values per pMHC. It needs to have the columns: allele, peptide, affinity")
arg_parser.add_argument("output_table", help="CSV file, that will be filled with ΔΔG values")



R = 8.31446261815324  # gas constant (J / (K * mol))
T = 298.15  # default temperature (K)

def get_dG(kd: float) -> float:

    # Kd is expected to be in nM
    return R * T * log(kd * 1e-9)


if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    args = arg_parser.parse_args()

    mutant_table = pandas.read_csv(args.mutants_table)

    o = {"mut_ID": [], "wt_ID": [], "predicted_ddG": [], "true_ddG": [], "wt_true_binder": [], "mut_true_binder": []}

    predictions_table = pandas.concat([pandas.read_csv(p) for p in args.predictions_table], axis=0)
    predictions_dict = {}
    for _, row in predictions_table.iterrows():
        key = (row.allele, row.peptide)
        predictions_dict[key] = row.affinity

    k_threshold = 500.0
    count_change_mismatch = 0
    count_total_mutants = mutant_table.shape[0]

    for _, row in mutant_table.iterrows():

        predicted_wt_affinity = predictions_dict[(row.allele, row.wt_peptide)]
        predicted_mut_affinity = predictions_dict[(row.allele, row.mut_peptide)]

        predicted_wt_k = 50000 ** (1.0 - predicted_wt_affinity)
        predicted_mut_k = 50000 ** (1.0 - predicted_mut_affinity)

        predicted_wt_dG = get_dG(predicted_wt_k)
        predicted_mut_dG = get_dG(predicted_mut_k)

        true_wt_k = row.wt_value
        true_mut_k = row.mut_value

        true_wt_dG = get_dG(true_wt_k)
        true_mut_dG = get_dG(true_mut_k)

        o["mut_ID"].append(row.mut_ID)
        o["wt_ID"].append(row.wt_ID)

        o["predicted_ddG"].append(predicted_mut_dG - predicted_wt_dG)
        o["true_ddG"].append(true_mut_dG - true_wt_dG)

        o["wt_true_binder"].append(true_wt_k < k_threshold)
        o["mut_true_binder"].append(true_mut_k < k_threshold)

        if (true_wt_k < k_threshold) == (true_mut_k < k_threshold):

            if (true_wt_k < k_threshold) != (predicted_wt_k < k_threshold):

                count_change_mismatch += 1

            elif (true_mut_k < k_threshold) != (predicted_mut_k < k_threshold):

                count_change_mismatch += 1

    pandas.DataFrame(o).to_csv(args.output_table, index=False)

    percentage_mismatch = 100.0 * count_change_mismatch / count_total_mutants
    _log.info(f" {percentage_mismatch:.1f} % binder/nonbinder mismatch")
