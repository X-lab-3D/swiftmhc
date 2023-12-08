#!/usr/bin/env python

import os
import sys
import re
from glob import glob
from tempfile import mkdtemp
from shutil import rmtree
from argparse import ArgumentParser

import h5py
import logging
import numpy
import pandas
from matplotlib import pyplot
from moviepy.editor import ImageSequenceClip


_log = logging.getLogger(__name__)


arg_parser = ArgumentParser(description="combine tcrspec csv.xz tables into a heatmap animation")
arg_parser.add_argument("hdf5_path", help="path to the hdf5 file, containing the data")
arg_parser.add_argument("data_type", help="type of data to take from hdf5 file: (loop_attention/protein_attention/cross_attention/other)")
arg_parser.add_argument("block", type=int, nargs="?", help="number of the block: (0/1/2/...)")
arg_parser.add_argument("head", type=int, nargs="?", help="number of the head: (0/1/2/...)")


def is_frame_id(s: str) -> bool:
    return re.match("\d+\.\d+", s) is not None


def get_epoch_number(frame_id: str) -> float:

    numbers = frame_id.split('.')

    epoch_number = int(numbers[0])
    batch_number = int(numbers[1])

    return float(epoch_number) + 0.001 * batch_number


if __name__ == "__main__":

    args = arg_parser.parse_args()
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    if args.block is not None and args.head is not None:

        output_name = os.path.basename(args.hdf5_path).replace(".hdf5","").replace("-animation", "") + f"-{args.data_type}-{args.block}-{args.head}"
    else:
        output_name = os.path.basename(args.hdf5_path).replace(".hdf5","").replace("-animation", "") + f"-{args.data_type}"

    dirname = os.path.basename(args.hdf5_path).replace(".hdf5","").replace("-animation", "")

    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    with h5py.File(args.hdf5_path, 'r') as hdf5_file:
        frame_ids = sorted(filter(is_frame_id, hdf5_file.keys()), key=get_epoch_number)

        png_files = []

        for frame_id in frame_ids:

            epoch_number = get_epoch_number(frame_id)

            if args.block is not None and args.head is not None:
                data = hdf5_file[f"{frame_id}/{args.data_type}"][args.block, args.head, ...]
            else:
                data = hdf5_file[f"{frame_id}/{args.data_type}"][:]

            figure = pyplot.figure()
            plot = figure.add_subplot()
            vmin=data.min().min()
            vmax=data.max().max()
            heatmap = plot.imshow(data, cmap="Greys", aspect="auto", vmin=vmin, vmax=vmax, interpolation="none")
            figure.colorbar(heatmap)
            pyplot.title(f"{output_name}, epoch:{epoch_number:.3f}")

            png_path = os.path.join(dirname, f"{output_name}-{frame_id}.png")

            figure.savefig(png_path, format="png")
            png_files.append(png_path)

            _log.debug(f"created {png_path}")

    clip = ImageSequenceClip(png_files, fps=10)
    clip.write_videofile(f"{output_name}.mp4", fps=15)
