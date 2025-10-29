#!/usr/bin/env python

from argparse import ArgumentParser

import pandas
from sklearn.metrics import roc_auc_score


arg_parser = ArgumentParser(description="calculates AUC from a SwiftMHC output CSV table")
arg_parser.add_argument("table_csv", help="must have columns 'true class' and 'output affinity'")


if __name__ == "__main__":

    args = arg_parser.parse_args()

    table = pandas.read_csv(args.table_csv)

    auc = roc_auc_score(table['true class'], table['output affinity'])

    print("ROC AUC is", round(auc, 2))
