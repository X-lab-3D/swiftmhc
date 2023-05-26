import logging
import random
import os
import csv
from tempfile import mkdtemp
from shutil import rmtree

import torch

from tcrspec.domain.amino_acid import amino_acids, amino_acids_by_name, AMINO_ACID_DIMENSION
from tcrspec.predictor import Predictor, _residue_weights_to_tables


_log = logging.getLogger(__name__)



def test_residue_weights_to_tables():

    n_complexes = 10
    n_pmhc_residues = 100
    n_cdr_residues = 20
    n_head = 8

    pmhc_sequence = torch.stack([torch.stack([random.choice(amino_acids).one_hot_code for _ in range(n_pmhc_residues)])
                                 for __ in range(n_complexes)])
    cdr_sequence = torch.stack([torch.stack([random.choice(amino_acids).one_hot_code for _ in range(n_cdr_residues)])
                                for __ in range(n_complexes)])

    weights = torch.rand(n_complexes, n_pmhc_residues, n_cdr_residues, n_head)

    directory_path = mkdtemp()

    allowed_names = list(amino_acids_by_name.keys())

    try:
        _residue_weights_to_tables(f"{directory_path}/test", pmhc_sequence, cdr_sequence, weights)

        file_name = os.listdir(directory_path)[0]

        file_path = os.path.join(directory_path, file_name)

        with open(file_path, 'rt') as f:
            r = csv.reader(f)

            header = next(r)

            for name in header[1:]:
                assert name in allowed_names, f"header does not match format: {header}"

            for row in r:

                name = row[0]
                assert name in allowed_names, f"row does not match format: {row}"

                for weights in row[1:]:
                    for weight in weights.split():
                        weight = float(weight)
                        assert weight >= 0.0 and weight <= 1.0, f"unexpected weight value {weight}"
    finally:
        rmtree(directory_path)


def test_predictor():

    n_complexes = 10
    n_pmhc_residues = 100
    n_cdr_residues = 20
    n_sequence_channels = AMINO_ACID_DIMENSION
    n_head = 8
    n_classes = 2

    module = Predictor(n_pmhc_residues, n_cdr_residues, n_head)

    pmhc_sequences = torch.rand(n_complexes, n_pmhc_residues, n_sequence_channels)
    cdr_sequences = torch.rand(n_complexes, n_cdr_residues, n_sequence_channels)

    phmc_transformations = torch.rand(n_complexes, n_pmhc_residues, 3, 4)

    predictions = module(cdr_sequences, pmhc_sequences, phmc_transformations,
                         create_weight_tables=True)

    assert predictions.shape[0] == n_complexes
