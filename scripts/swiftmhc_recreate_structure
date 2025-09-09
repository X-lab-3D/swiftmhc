#!/usr/bin/env python

import os
from argparse import ArgumentParser
import h5py
import torch
from Bio.PDB.PDBIO import PDBIO
from swiftmhc.tools.pdb import recreate_structure


arg_parser = ArgumentParser("recreate a structure from a hdf5 dataset file")
arg_parser.add_argument("dataset_file")
arg_parser.add_argument("structure_id")


if __name__ == "__main__":
    args = arg_parser.parse_args()

    path = os.path.splitext(args.dataset_file)[0] + f"-{args.structure_id}.pdb"

    with h5py.File(args.dataset_file, "r") as f5:
        data = f5[args.structure_id]

        selection = [
            (
                "M",
                torch.tensor(data["protein/residue_numbers"]),
                torch.tensor(data["protein/aatype"]),
                torch.tensor(data["protein/atom14_gt_positions"]),
                torch.tensor(data["protein/atom14_gt_exists"]),
            )
        ]
        if "atom14_gt_positions" in data["peptide"]:
            selection.append(
                (
                    "P",
                    torch.tensor(data["peptide/residue_numbers"]),
                    torch.tensor(data["peptide/aatype"]),
                    torch.tensor(data["peptide/atom14_gt_positions"]),
                    torch.tensor(data["peptide/atom14_gt_exists"]),
                )
            )

        structure = recreate_structure(args.structure_id, selection)

        pdbio = PDBIO()
        pdbio.set_structure(structure)
        pdbio.save(path)

        print("saved to", path)
