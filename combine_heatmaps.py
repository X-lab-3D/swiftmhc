#!/usr/bin/env python

import os
import sys
import csv
from tempfile import mkdtemp
from shutil import rmtree

import numpy
import pandas
from matplotlib import pyplot
from moviepy.editor import ImageSequenceClip


def chronological(path: str) -> float:

    # train-epoch-11-batch-80-target-NONBINDING-crossweights-LLLENKSLT.csv
    filename = os.path.basename(path)

    words = filename.split('-')

    epoch_number = int(words[2])
    batch_number = int(words[4])

    return float(epoch_number) + 0.001 * batch_number


if __name__ == "__main__":

    table_directory_path = sys.argv[1]

    storage_directory_path = mkdtemp()
    png_files = []
    try:
        for filename in os.listdir(table_directory_path):
            if "crossweights" in filename and filename.endswith(".csv"):

                table_path = os.path.join(table_directory_path, filename)

                data = pandas.read_csv(table_path, sep=',', index_col=0, header=0)

                figure = pyplot.figure()
                plot = figure.add_subplot()
                heatmap = plot.imshow(data, cmap="Greys", aspect="auto", vmin=0.0, vmax=0.3)
                figure.colorbar(heatmap)
                pyplot.title(filename)

                png_path = os.path.join(storage_directory_path, filename.replace(".csv", ".png"))

                figure.savefig(png_path, format="png")
                png_files.append(png_path)

        clip = ImageSequenceClip(sorted(png_files, key=chronological), fps=10)
        clip.write_videofile(f"{table_directory_path}/cross-attention.mp4", fps=10)
    finally:
        rmtree(storage_directory_path)
