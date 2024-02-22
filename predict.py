#!/usr/bin/env python

import os
import logging
from argparse import ArgumentParser
from multiprocessing import Pool
from typing import Dict, List, Tuple

import pandas
import torch

from Bio.PDB.PDBIO import PDBIO

from openfold.config import config as openfold_config

from swiftmhc.tools.pdb import recreate_structure
from swiftmhc.models.types import ModelType
from swiftmhc.modules.predictor import Predictor
from swiftmhc.dataset import ProteinLoopDataset


_log = logging.getLogger(__name__)


PEPTIDE_MAXLEN = 16
PROTEIN_MAXLEN = 200


arg_parser = ArgumentParser(description="predict peptide structures and binding affinity from peptide sequence and MHC structures")
arg_parser.add_argument("model_path", help="pretrained model")
arg_parser.add_argument("table_path", help="table containing the columns: 'peptide' and 'allele'")
arg_parser.add_argument("hdf5_path", help="an hdf5 file containing the preprocessed mhc structures, identified by allele")
arg_parser.add_argument("output_directory", help="directory for the output files: *.pdb and *.csv")
arg_parser.add_argument("--batch-size", "-b", type=int, default=64, help="number of simultaneous complexes to predict in one batch")
arg_parser.add_argument("--num-workers", "-w", type=int, default=5, help="number of simulteneous data readers")
arg_parser.add_argument("--num-builders", "-B", type=int, default=64, help="number of simultaneous structure builders")


def create_dataset(table_path: str, hdf5_path: str, device: torch.device) -> ProteinLoopDataset:

    table = pandas.read_csv(table_path)
    pairs = []
    for _, row in table.iterrows():
        pairs.append((row["peptide"], row["allele"]))

    dataset = ProteinLoopDataset(hdf5_path, device, PEPTIDE_MAXLEN, PROTEIN_MAXLEN, pairs=pairs)
    return dataset


def save_structure(path: str, data: List[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]]):

    structure = recreate_structure(path, data)

    pdbio = PDBIO()
    pdbio.set_structure(structure)
    pdbio.save(path)


def on_error(e):
    "callback function, send the error to the logs"
    _log.error(str(e))


def store_output(
    pool: Pool,
    directory_path: str,
    batch: Dict[str, torch.Tensor],
    output: Dict[str, torch.Tensor]
):
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)

    # save structures
    for index, id_ in enumerate(batch["ids"]):

        pdb_path = os.path.join(directory_path, f"{id_}.pdb")

        peptide_residue_numbers = batch["peptide_residue_numbers"].cpu()
        peptide_sequence_onehot = batch["peptide_sequence_onehot"].cpu()
        peptide_atom14_positions = output["final_positions"].cpu()

        protein_residue_numbers = batch["protein_residue_numbers"].cpu()
        protein_sequence_onehot = batch["protein_sequence_onehot"].cpu()
        protein_atom14_gt_positions = batch["protein_atom14_gt_positions"].cpu()

        pool.apply_async(
            save_structure,
            (
                pdb_path,
                [("P", peptide_residue_numbers[index], peptide_sequence_onehot[index], peptide_atom14_positions[index]),
                 ("M", protein_residue_numbers[index], protein_sequence_onehot[index], protein_atom14_gt_positions[index])]
            ),
            error_callback=on_error,
        )

    # save affinity/class
    table_path = os.path.join(directory_path, "results.csv")
    data_dict = {
        "ID": batch["ids"],
        "allele": batch["allele"],
        "peptide": batch["peptide"],
    }
    if "affinity" in output:
        data_dict["affinity"] = output["affinity"]

    elif "class" in output:
        data_dict["class"] = output["class"]

    if os.path.isfile(table_path):
        data = pandas.read_csv(table_path)
        data = pandas.concat((data, pandas.DataFrame(data_dict)), axis=0)
    else:
        data = pandas.DataFrame(data_dict)
    data.to_csv(table_path, index=False)



if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    args = arg_parser.parse_args()

    dataset = create_dataset(args.table_path, args.hdf5_path, device)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=ProteinLoopDataset.collate,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # Set up the model
    model = Predictor(PEPTIDE_MAXLEN, PROTEIN_MAXLEN, ModelType.REGRESSION, openfold_config.model)
    model = torch.nn.DataParallel(model)
    model.to(device=device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    with Pool(args.num_builders) as builder_pool:
        with torch.no_grad():
            for batch in data_loader:
                output = model(batch)

                store_output(builder_pool, args.output_directory, batch, output)
