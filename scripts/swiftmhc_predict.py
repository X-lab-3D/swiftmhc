#!/usr/bin/env python

import logging
import os
import sys
from argparse import ArgumentParser
from multiprocessing import Pool
from multiprocessing.pool import Pool as PoolType
import openmm.app
import pandas
import torch
from Bio.PDB.PDBIO import PDBIO
from openfold.np.residue_constants import restype_atom14_mask
from swiftmhc.config import config
from swiftmhc.dataset import ProteinLoopDataset
from swiftmhc.modules.predictor import Predictor
from swiftmhc.preprocess import affinity_binding_threshold
from swiftmhc.tools.md import build_modeller
from swiftmhc.tools.md import minimize
from swiftmhc.tools.pdb import recreate_structure


_log = logging.getLogger(__name__)


PEPTIDE_MAXLEN = 16
PROTEIN_MAXLEN = 200


arg_parser = ArgumentParser(
    description="predict peptide structures and binding affinity from peptide sequence and MHC structures"
)
arg_parser.add_argument("model_path", help="pretrained model")
arg_parser.add_argument("table_path", help="table containing the columns: 'peptide' and 'allele'")
arg_parser.add_argument(
    "hdf5_path",
    help="an hdf5 file containing the preprocessed mhc structures, identified by allele",
)
arg_parser.add_argument("output_directory", help="directory for the output files: *.pdb and *.csv")
arg_parser.add_argument(
    "--with-energy-minimization",
    help="include structure OpenMM energy minimization",
    action="store_const",
    const=True,
    default=False,
)
arg_parser.add_argument(
    "--batch-size",
    "-b",
    type=int,
    default=64,
    help="number of simultaneous complexes to predict in one batch",
)
arg_parser.add_argument(
    "--num-workers", "-w", type=int, default=5, help="number of simulteneous data readers"
)
arg_parser.add_argument(
    "--num-builders",
    "-B",
    type=int,
    default=1,
    help="number of simultaneous structure builders, set to 0 to disable structure prediction",
)


def remove_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Paralellisation adds a 'module.' prefix in the state's keys.
    Remove those prefixes here.
    """
    d = {}
    for key, value in state.items():
        if key.startswith("module."):
            new_key = key[7:]

            if new_key in d:
                raise ValueError(f"duplicate key: {new_key}")

            d[new_key] = value
        else:
            d[key] = value

    return d


def create_dataset(
    table_path: str, hdf5_path: str, device: torch.device, float_dtype: torch.dtype
) -> ProteinLoopDataset:
    table = pandas.read_csv(table_path)
    pairs = []
    for _, row in table.iterrows():
        pairs.append((row["peptide"], row["allele"]))

    dataset = ProteinLoopDataset(
        hdf5_path, device, float_dtype, PEPTIDE_MAXLEN, PROTEIN_MAXLEN, pairs=pairs
    )
    return dataset


def output_structure(
    path: str,
    data: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    minimize_energy: bool,
):
    if minimize_energy:
        modeller = build_modeller(data)
        modeller = minimize(modeller)

        with open(path, "wt") as f:
            openmm.app.PDBFile.writeFile(
                modeller.topology, modeller.getPositions(), f, keepIds=True
            )
    else:
        structure = recreate_structure(path, data)

        io = PDBIO()
        io.set_structure(structure)
        io.save(path)


def store_output(
    pool: PoolType | None,
    directory_path: str,
    batch: dict[str, torch.Tensor],
    output: dict[str, torch.Tensor],
    minimize_energy: bool,
):
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)

    # save structures
    for index, id_ in enumerate(batch["ids"]):
        pdb_name = id_.replace("*", "x").replace(":", "_")
        pdb_path = os.path.join(directory_path, f"{pdb_name}.pdb")

        peptide_residue_numbers = batch["peptide_residue_numbers"].cpu()
        peptide_aatype = batch["peptide_aatype"].cpu()

        peptide_atom14_positions = output["final_positions"].cpu()
        peptide_mask = batch["peptide_self_residues_mask"].cpu()
        peptide_atom14_mask = (
            peptide_mask.new_tensor(restype_atom14_mask[peptide_aatype]) * peptide_mask[..., None]
        )

        protein_residue_numbers = batch["protein_residue_numbers"].cpu()
        protein_aatype = batch["protein_aatype"].cpu()
        protein_atom14_positions = batch["protein_atom14_gt_positions"].cpu()
        protein_atom14_exists = batch["protein_atom14_gt_exists"].cpu()

        if pool is not None and (
            "class" in output
            and output["class"][index] > 0
            or "affinity" in output
            and output["affinity"][index] > affinity_binding_threshold
        ):
            _log.info(f"add to build pool: {pdb_path}")

            # found a binder case, store the model
            pool.apply_async(
                output_structure,
                (
                    pdb_path,
                    [
                        (
                            "P",
                            peptide_residue_numbers[index],
                            peptide_aatype[index],
                            peptide_atom14_positions[index],
                            peptide_atom14_mask[index],
                        ),
                        (
                            "M",
                            protein_residue_numbers[index],
                            protein_aatype[index],
                            protein_atom14_positions[index],
                            protein_atom14_exists[index],
                        ),
                    ],
                    minimize_energy,
                ),
                error_callback=lambda e: _log.exception(f"on {pdb_path}"),
            )

    # save affinity/class
    table_path = os.path.join(directory_path, "results.csv")
    data_dict = {
        "ID": batch["ids"],
        "allele": batch["allele"],
        "peptide": batch["peptide"],
    }
    if "affinity" in output:
        data_dict["affinity"] = output["affinity"].cpu()
        data_dict["class"] = (data_dict["affinity"].cpu() > affinity_binding_threshold).to(
            dtype=torch.int
        )

    elif "class" in output:
        data_dict["class"] = output["class"].cpu()

    if os.path.isfile(table_path):
        data = pandas.read_csv(table_path)
        data = pandas.concat((data, pandas.DataFrame(data_dict)), axis=0)
    else:
        data = pandas.DataFrame(data_dict)
    data.to_csv(table_path, index=False)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    float_dtype = torch.float32

    args = arg_parser.parse_args()

    dataset = create_dataset(args.table_path, args.hdf5_path, torch.device("cpu"), float_dtype)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=ProteinLoopDataset.collate,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )

    # required for CUDA & multiprocessing
    torch.multiprocessing.set_start_method("spawn")

    # Set up the model
    model = Predictor(config)
    model.to(device=device)
    model.load_state_dict(remove_module_prefix(torch.load(args.model_path, map_location=device)))
    model.to(dtype=float_dtype)
    model = torch.nn.DataParallel(model)
    model.eval()

    if args.num_builders > 0:
        with Pool(args.num_builders) as builder_pool:
            with torch.no_grad():
                for batch in data_loader:
                    # Transfer batch to device
                    batch = {
                        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }

                    output = model(batch)

                    store_output(
                        builder_pool,
                        args.output_directory,
                        batch,
                        output,
                        args.with_energy_minimization,
                    )

            builder_pool.close()
            builder_pool.join()
    else:
        with torch.no_grad():
            for batch in data_loader:
                # Transfer batch to device
                batch = {
                    k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                output = model(batch)

                store_output(None, args.output_directory, batch, output, False)
