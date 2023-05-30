#!/usr/bin/env python

from argparse import ArgumentParser
import os
import re
from typing import List
import csv
import logging
import sys

import pandas
import numpy


arg_parser = ArgumentParser(description="Convert cross attention weight tables into a ranking for anchor points and two pockets")
arg_parser.add_argument("crossweights_directory", help="directory where to find the crossweight tables")
arg_parser.add_argument("binding_affinities_path", help="file where the binding affinities are stored")
arg_parser.add_argument("--binding", "-b", dest="binding", action="store_true")
arg_parser.add_argument("--nonbinding", "-n", dest="binding", action="store_false")
arg_parser.set_defaults(binding=None)


pocket1_resnums = numpy.array([7, 9, 45, 63, 66, 67, 70, 99, 159, 167])
pocket2_resnums = numpy.array([77, 80, 81, 84, 95, 116, 123, 124, 142, 143, 146, 147])


EPOCH_PATTERN = re.compile(r"epoch-(\d+)")


_log = logging.getLogger(__name__)


def find_crossweights(directory_path: str):
    crossweight_paths = []
    for name in os.listdir(directory_path):
        path = os.path.join(directory_path, name)

        if os.path.isdir(path):
            crossweight_paths += find_crossweights(path)

        elif name.endswith(".csv"):
            match = EPOCH_PATTERN.search(name)
            if match is None:
                continue

            epoch_index = int(match.group(1))

            if epoch_index >= 20:
                crossweight_paths.append(path)

    return crossweight_paths


def get_epitope(path: str):
    name = os.path.splitext(os.path.basename(path))[0]
    return name.split('-')[-1]


def get_weights(path: str):
    table = pandas.read_csv(path, sep=',', header=0, index_col=0)
    return table.values


TENSOR_PATTERN = re.compile(r"\[([0-9\.\-\+eE ]+\,[0-9\.\-\+eE ]+)\]")


def get_output(crossweight_path: str):

    # train-epoch-3-batch-54-target-0.0-crossweights-TTELRTFSI.csv
    # train-epoch-3-batch-54-target-0.0-TTELRTFSI.txt

    txt_path = crossweight_path.replace("-crossweights", "").replace(".csv", ".txt")

    with open(txt_path, 'rt') as f:
        line = f.readline()

    match = TENSOR_PATTERN.search(line)
    if match is None:
        raise ValueError(f"no tensor in \"{line}\"")

    return [float(s) for s in match.group(1).split(',')]


def get_binding_affinities(table_path: str):

    return pandas.read_csv(table_path, sep=',', header=0, index_col=0)


def get_sorted_resnums(r: List[float]) -> List[int]:

    numbers = list(range(1, len(r) + 1))

    return sorted(numbers, key=lambda n: r[n - 1],
                  reverse=True)

def count_ranked_first(r: numpy.ndarray, resnum: int) -> int:

    count = 0
    for index in range(r.shape[0]):

        sorted_ = get_sorted_resnums(r[index].tolist())

        if sorted_[0] == resnum:
            count += 1

    return count


def output_table(rank_path: str,
                 selection: numpy.ndarray,
                 epitopes: List[str],
                 outputs: numpy.ndarray,
                 affinities: numpy.ndarray,
                 pocket1_r: numpy.ndarray,
                 pocket2_r: numpy.ndarray):

    with open(rank_path, 'wt') as f:
        w = csv.writer(f, delimiter=',')

        w.writerow(["epitope", "output[0]", "output[1]",
                    "Kd (nM)", "pocket 1 order", "pocket 2 order"])
        for i in selection:
            w.writerow([epitopes[i], outputs[i, 0], outputs[i, 1],
                        affinities[i],
                        " ".join([str(n) for n in get_sorted_resnums(pocket1_r[i].tolist())]),
                        " ".join([str(n) for n in get_sorted_resnums(pocket2_r[i].tolist())])])



if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    args = arg_parser.parse_args()

    _log.debug("finding crossweights..")

    crossweight_paths = find_crossweights(args.crossweights_directory)
    _log.debug(f"paths = [{crossweight_paths[0]}, {crossweight_paths[1]}, ...")

    # remove double epitopes:
    paths_per_epitope = {}
    for crossweight_path in crossweight_paths:
        epitope = get_epitope(crossweight_path)
        paths_per_epitope[epitope] = crossweight_path
    crossweight_paths = list(paths_per_epitope.values())
    epitopes = list(paths_per_epitope.keys())

    _log.debug(f"epitopes = [{epitopes[0]}, {epitopes[1]}, ...")
    _log.debug(f"{len(epitopes)} crossweights found, now parsing..")

    # [n_peptides, n_struct_res, n_loop_res]
    crossweights = numpy.stack([get_weights(crossweight_path) for crossweight_path in crossweight_paths], axis=0)

    _log.debug("loading outputs..")

    # [n_peptides, 2]
    outputs = numpy.array([get_output(crossweight_path) for crossweight_path in crossweight_paths])

    _log.debug("loading affinities..")

    # [n_peptides, 1]
    binding_affinities = get_binding_affinities(args.binding_affinities_path)
    binding_affinities = binding_affinities.set_index("peptide")
    binding_affinities = binding_affinities.groupby("peptide")["measurement_value"].mean().loc[epitopes]
    binding_affinities = binding_affinities.to_numpy().reshape(len(epitopes)).astype("float")

    # selection is a list of indexes, pointing to the peptides we want to include.
    if args.binding is None:
        selection = numpy.arange(0, binding_affinities.shape[0], 1)
        rank_path = os.path.join(args.crossweights_directory, "rankings.csv")

    elif args.binding:
        selection = numpy.nonzero(numpy.logical_and(binding_affinities < 500, outputs[:, 0] < outputs[:, 1]))[0]
        rank_path = os.path.join(args.crossweights_directory, "binding-rankings.csv")
    else:
        selection = numpy.nonzero(numpy.logical_and(binding_affinities >= 500, outputs[:, 0] > outputs[:, 1]))[0]
        rank_path = os.path.join(args.crossweights_directory, "nonbinding-rankings.csv")

    _log.debug("summing pocket weights..")

    # sum over the pocket resdiues
    pocket1_w = numpy.sum(crossweights[:, pocket1_resnums - 1, :], axis=1)
    pocket2_w = numpy.sum(crossweights[:, pocket2_resnums - 1, :], axis=1)

    _log.debug("calculating p-values..")

    # calculate p = w_i / sum_i_9(w_i)
    pocket1_p = pocket1_w / numpy.expand_dims(numpy.sum(pocket1_w, axis=1), axis=1)
    pocket2_p = pocket2_w / numpy.expand_dims(numpy.sum(pocket2_w, axis=1), axis=1)

    _log.debug("calculating ranks..")

    # calculate the values on which we base the ranking:
    # for peptide k of a total N peptides
    # r = log(p_k / mean_k_N(p_k)
    pocket1_r = numpy.log(pocket1_p / numpy.expand_dims(numpy.mean(pocket1_p, axis=1), axis=1))
    pocket2_r = numpy.log(pocket2_p / numpy.expand_dims(numpy.mean(pocket2_p, axis=1), axis=1))

    _log.debug(f"selection {selection.shape} {selection.dtype}")
    _log.debug(f"epitopes: {len(epitopes)}")
    _log.debug(f"outputs: {outputs.shape} {outputs.dtype}")
    _log.debug(f"affinities: {binding_affinities.shape} {binding_affinities.dtype}")
    _log.debug(f"pocket1 ranges: {pocket1_r.shape} {pocket1_r.dtype}")
    _log.debug(f"pocket2 ranges: {pocket2_r.shape} {pocket2_r.dtype}")

    _log.debug("outputting table..")

    output_table(rank_path, selection, epitopes, outputs, binding_affinities,
                 pocket1_r, pocket2_r)

    print("Written to:", rank_path)

    # Output percentages ranked 2 and 9 in front.
    count_pocket1 = count_ranked_first(pocket1_r[selection], 2)
    count_pocket2 = count_ranked_first(pocket2_r[selection], 9)

    percent_pocket1 = (100.0 * count_pocket1) / pocket1_r[selection].shape[0]
    percent_pocket2 = (100.0 * count_pocket2) / pocket2_r[selection].shape[0]

    print(f"{percent_pocket1} % of peptides have residue 2 ranked first on pocket 1")
    print(f"{percent_pocket2} % of peptides have residue 9 ranked first on pocket 2")
