#!/usr/bin/env python

import os
import sys
import csv
from glob import glob
from tempfile import mkdtemp
from shutil import rmtree
from argparse import ArgumentParser

import logging
import numpy
import pandas
from matplotlib import pyplot
from moviepy.editor import ImageSequenceClip


_log = logging.getLogger(__name__)


arg_parser = ArgumentParser(description="combine tcrspec csv tables into a heatmap animation")
arg_parser.add_argument("prefix", help="prefix used in the csv filenames")


def get_epoch_number(path: str) -> float:

    # BA-55224-20.160_h0.csv
    filename = os.path.basename(path)

    word = filename.split('-')[-1]
    numbers = word.split('.')

    epoch_number = int(numbers[0])
    batch_number = int(numbers[1])

    return float(epoch_number) + 0.001 * batch_number


if __name__ == "__main__":

    args = arg_parser.parse_args()
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    table_paths = glob(f"{args.prefix}-*.*.csv")
    table_paths = sorted(table_paths, key=get_epoch_number)
    if len(table_paths) == 0:
        raise FileNotFoundError(f"no csv files for prefix {args.prefix}")

    storage_directory_path = mkdtemp()
    png_files = []

    for table_path in table_paths:
        epoch_number = get_epoch_number(table_path)

        data = pandas.read_csv(table_path, sep=',', index_col=0, header=0)

        figure = pyplot.figure()
        plot = figure.add_subplot()
        vmin=data.min().min()
        vmax=data.max().max()
        heatmap = plot.imshow(data, cmap="Greys", aspect="auto", vmin=vmin, vmax=vmax)
        figure.colorbar(heatmap)
        pyplot.title(f"{args.prefix}, epoch:{epoch_number:.3f}")

        png_path = table_path.replace(".csv", ".png")

        figure.savefig(png_path, format="png")
        png_files.append(png_path)

        _log.debug(f"created {png_path}")

    clip = ImageSequenceClip(png_files, fps=10)
    clip.write_videofile(f"{args.prefix}.mp4", fps=15)
