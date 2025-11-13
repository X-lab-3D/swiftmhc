#!/usr/bin/env python

from argparse import ArgumentParser
import sys
import logging

import pandas


arg_parser = ArgumentParser(description="find peptide single point mutations in a IEDB dataset by comparing all against all")
arg_parser.add_argument("input_table", nargs="+", help="CSV with IEDB data. It must have the columns: ID, allele, peptide, measurement_value (as in IEDB, the Kd or IC50), cluster")
arg_parser.add_argument("output_table", help="CSV where the output mutant data must be stored")


_log = logging.getLogger(__name__)


if __name__ == "__main__":

    args = arg_parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    input_table = None
    for path in args.input_table:

        table = pandas.read_csv(path)
        if "measurement_value" not in table:
            table["measurement_value"] = "?"

        if input_table is None:
            input_table = table
        else:
            input_table = pandas.concat((input_table, table), axis=0)

    output_table = pandas.DataFrame({
        "wt_ID": [],
        "wt_peptide": [],
        "wt_value": [],
        "wt_cluster": [],
        "allele": [],
        "mut_ID": [],
        "mut_peptide": [],
        "mut_value": [],
        "mut_cluster": [],
    })

    for i, row_i in input_table.iterrows():

        id_i = row_i["ID"]
        allele_i = row_i["allele"]
        peptide_i = row_i['peptide']
        value_i = row_i["measurement_value"]
        cluster_i = row_i["cluster"]

        for j, row_j in input_table.iterrows():

            id_j = row_j["ID"]
            allele_j = row_j["allele"]
            peptide_j = row_j['peptide']
            value_j = row_j["measurement_value"]
            cluster_j = row_j["cluster"]

            if len(peptide_i) == len(peptide_j) and allele_i == allele_j:

                mismatches = 0
                for k in range(len(peptide_i)):

                    if peptide_i[k] != peptide_j[k]:
                        mismatches += 1

                if mismatches == 1:

                    output_row = pandas.DataFrame({
                        "wt_ID": [id_i],
                        "wt_peptide": [peptide_i],
                        "wt_value": [value_i],
                        "wt_cluster": [cluster_i],
                        "allele": [allele_i],
                        "mut_ID": [id_j],
                        "mut_peptide": [peptide_j],
                        "mut_value": [value_j],
                        "mut_cluster": [cluster_j],
                    })

                    output_table = pandas.concat((output_table, output_row), axis=0)

    output_table.to_csv(args.output_table, index=False)
