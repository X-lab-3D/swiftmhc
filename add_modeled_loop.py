#!/usr/bin/env python

import re
import sys
import logging
from argparse import ArgumentParser

from pymol import cmd, cgo

arg_parser = ArgumentParser(description="visualize loop positions data in a pymol scene")
arg_parser.add_argument("txt_path", help="where the loop data is stored")
arg_parser.add_argument("input_session", help="pymol session that should be modified")


_log = logging.getLogger(__name__)


POS_PATTERN = re.compile(r"\[(.*)\,(.*)\,(.*)\]")


def read_txt(path: str):

    with open(path, 'rt') as f:

        header = f.readline().split()
        loop_seq = header[2]

        loop_positions = {}
        for line in f:
            if line.startswith("loop residue "):
                row = line.split()

                residue_index = int(row[2])

                pos_match = POS_PATTERN.search(line)
                pos = [float(pos_match.group(1)),
                       float(pos_match.group(2)),
                       float(pos_match.group(3))]

                loop_positions[residue_index] = pos

    return (loop_seq, loop_positions)


if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    args = arg_parser.parse_args()

    loop_seq, loop_positions = read_txt(args.txt_path)

    cmd.load(args.input_session)

    loop_len = len(loop_seq)

    for residue_index, pos in loop_positions.items():

        letter = loop_seq[residue_index]

        loop_frac = (residue_index / loop_len)

        cgo_obj = [cgo.COLOR, 0.0, 1.0 - loop_frac, loop_frac,
                   cgo.SPHERE] + pos + [1.0]

        cmd.load_cgo(cgo_obj, f"{letter}{residue_index}")

    cmd.center("all")
    cmd.zoom("all")

    file_prefix = args.txt_path.replace(".txt", "-loopprediction")
    session_path = file_prefix + ".pse"
    png_path = file_prefix + ".png"

    cmd.save(session_path)
    cmd.png(png_path)
    _log.info(session_path)
    _log.info(png_path)
