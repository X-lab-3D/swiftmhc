#!/usr/bin/env python

import os
import sys
import csv
from glob import glob
from tempfile import mkdtemp
from shutil import rmtree

import numpy
import pandas
from matplotlib import pyplot
from moviepy.editor import ImageSequenceClip


def get_epoch_number(path: str) -> float:

    # BA-55224-20.160_h0.csv
    filename = os.path.basename(path)

    word = filename.split('-')[-1].split('_h')[0]
    numbers = word.split('.')

    epoch_number = int(numbers[0])
    batch_number = int(numbers[1])

    return float(epoch_number) + 0.001 * batch_number


if __name__ == "__main__":

    complex_id = sys.argv[1]
    head_id = sys.argv[2]

    table_paths = glob(f"{complex_id}-*.*_h{head_id}.csv")
    table_paths = sorted(table_paths, key=get_epoch_number)

    storage_directory_path = mkdtemp()
    png_files = []

    for table_path in table_paths:

        data = pandas.read_csv(table_path, sep=',', index_col=0, header=0)

        figure = pyplot.figure()
        plot = figure.add_subplot()
        heatmap = plot.imshow(data, cmap="Greys", aspect="auto", vmin=0.0, vmax=0.3)
        figure.colorbar(heatmap)
        pyplot.title(filename)

        png_path = table_path.replace(".csv", ".png")

        figure.savefig(png_path, format="png")
        png_files.append(png_path)

    clip = ImageSequenceClip(png_files, fps=10)
    clip.write_videofile(f"{complex_id}_{head_id}.mp4", fps=15)
