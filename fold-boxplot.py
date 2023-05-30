#!/usr/bin/env python

import os
from argparse import ArgumentParser

from matplotlib import pyplot
import pandas


arg_parser = ArgumentParser(description="make boxplots for the folds")
arg_parser.add_argument("directories", help="where the folds are stored", nargs="+")


metrics_names = ["test mcc", "test sensitivity", "test specificity", "test accuracy"]


if __name__ == "__main__":

    args = arg_parser.parse_args()

    directory_names = [os.path.basename(directory_path)
                       for directory_path in args.directories]

    data_frames = {}

    for metrics_name in metrics_names:

        phase_name, metrics_type_name = metrics_name.split()

        data = pandas.DataFrame([], dtype='float', columns=directory_names)

        for directory_path in args.directories:

            directory_name = os.path.basename(directory_path)

            for fold_name in os.listdir(directory_path):

                metrics_path = os.path.join(directory_path, fold_name, "metrics.csv")
                if os.path.isfile(metrics_path):

                    metrics = pandas.read_csv(metrics_path, sep=',', header=0, index_col=0)

                    data.at[fold_name, directory_name] = float(metrics[metrics_name][0].item())

        figure = pyplot.figure()
        boxplot = data.boxplot(column=directory_names)

        boxplot_path = os.path.join(f"test-{metrics_type_name}-boxplot.png")
        figure.savefig(boxplot_path, format="png")
        print("saved to:", boxplot_path)
