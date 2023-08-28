import os
import sys
import csv
import logging
from typing import Optional, List, Tuple, Dict, Union
from math import log
from enum import Enum

import h5py
from pdb2sql import pdb2sql
import torch
from torch.utils.data import Dataset
from torch.nn.functional import pad
from sklearn.decomposition import PCA

from openfold.utils.rigid_utils import Rigid

from .models.types import ModelType
from .models.data import TensorDict
from .models.residue import Residue
from .models.complex import ComplexClass, ComplexTableEntry, ComplexDataEntry, StructureDataEntry
from .domain.amino_acid import amino_acids_by_letter, amino_acids_by_one_hot_index, AMINO_ACID_DIMENSION
from .tools.pdb import get_selected_residues, get_residue_transformations, get_residue_proximities
from .preprocess import PREPROCESS_KD_NAME, PREPROCESS_PROTEIN_NAME, PREPROCESS_LOOP_NAME
from .modules.sequence_encoding import mask_loop_left_center_right


_log = logging.getLogger(__name__)


class ProteinLoopDataset(Dataset):
    def __init__(self, hdf5_path: str, device: torch.device, model_type: ModelType, loop_maxlen: int, protein_maxlen: int):
        self._hdf5_path = hdf5_path
        self._device = device
        self._loop_maxlen = loop_maxlen
        self._protein_maxlen = protein_maxlen
        self._model_type = model_type

        with h5py.File(self._hdf5_path, 'r') as hdf5_file:
            self._entry_names = list(hdf5_file.keys())

    def __len__(self) -> int:
        return len(self._entry_names)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:

        entry_name = self._entry_names[index]

        return self.get_entry(entry_name)

    def has_entry(self,  entry_name: str) -> bool:
        return entry_name in self._entry_names

    def get_entry(self, entry_name: str) -> Dict[str, torch.Tensor]:

        result = {}

        with h5py.File(self._hdf5_path, 'r') as hdf5_file:
            entry_group = hdf5_file[entry_name]

            result["ids"] = entry_name

            if PREPROCESS_KD_NAME in entry_group:
                result["kd"] = torch.tensor(entry_group[PREPROCESS_KD_NAME][()], device=self._device, dtype=torch.float)
                result["affinity"] = 1.0 - torch.log(result["kd"]) / log(50000)
                result["class"] = torch.tensor(entry_group[PREPROCESS_KD_NAME][()] < 500.0, device=self._device, dtype=torch.long)

            for prefix, max_length in [(PREPROCESS_PROTEIN_NAME, self._protein_maxlen),
                                       (PREPROCESS_LOOP_NAME, self._loop_maxlen)]:

                aatype_data = entry_group[prefix]["aatype"][:]
                length = aatype_data.shape[0]
                if length < 3:
                    raise ValueError(f"{entry_name} {prefix} length is {length}")

                # For the protein, put all residues leftmost
                index = torch.zeros(max_length, device=self._device, dtype=torch.bool)
                index[:length] = True

                # For the loop, put residues partly leftmost, partly centered, partly rightmost
                if prefix == PREPROCESS_LOOP_NAME:
                    index = mask_loop_left_center_right(length, max_length)

                result[f"{prefix}_aatype"] = torch.zeros(max_length, device=self._device, dtype=torch.long)
                result[f"{prefix}_aatype"][index] = torch.tensor(aatype_data, device=self._device, dtype=torch.long)

                for interfix in ["self", "cross"]:
                    result[f"{prefix}_{interfix}_residues_mask"] = torch.zeros(max_length, device=self._device, dtype=torch.bool)
                    key = "{interfix}_residues_mask"

                    if key in entry_group[prefix]:
                        mask_data = entry_group[prefix][key][:]
                        result[f"{prefix}_{interfix}_residues_mask"][index] = mask_data
                    else:
                        # If no mask, then set all present residues to True.
                        result[f"{prefix}_{interfix}_residues_mask"][index] = True

                result[f"{prefix}_residue_index"] = torch.arange(0, max_length, 1, device=self._device, dtype=torch.long)
                result[f"{prefix}_residue_numbers"] = torch.zeros(max_length, dtype=torch.int, device=self._device)
                result[f"{prefix}_residue_numbers"][index] = torch.tensor(entry_group[prefix]["residue_numbers"][:], dtype=torch.int, device=self._device)

                residx_atom14_to_atom37_data = entry_group[prefix]["residx_atom14_to_atom37"][:]
                result[f"{prefix}_residx_atom14_to_atom37"] = torch.zeros((max_length, residx_atom14_to_atom37_data.shape[1]), device=self._device, dtype=torch.long)
                result[f"{prefix}_residx_atom14_to_atom37"][index] = torch.tensor(residx_atom14_to_atom37_data, device=self._device, dtype=torch.long)

                result[f"{prefix}_sequence_onehot"] = torch.zeros((max_length, 32), device=self._device, dtype=torch.float)
                t = torch.tensor(entry_group[prefix]["sequence_onehot"][:], device=self._device, dtype=torch.float)
                result[f"{prefix}_sequence_onehot"][:t.shape[0], :t.shape[1]] = t

                for field_name in ["backbone_rigid_tensor",
                                   "torsion_angles_sin_cos", "alt_torsion_angles_sin_cos", "torsion_angles_mask",
                                   "atom14_gt_exists", "atom14_gt_positions", "atom14_alt_gt_positions",
                                   "all_atom_mask", "all_atom_positions"]:

                    data = entry_group[prefix][field_name][:]
                    length = data.shape[0]
                    t = torch.zeros([max_length] + list(data.shape[1:]), device=self._device, dtype=torch.float)
                    t[index] = torch.tensor(data, device=self._device, dtype=torch.float)

                    result[f"{prefix}_{field_name}"] = t

            prox_data = entry_group[PREPROCESS_PROTEIN_NAME]["proximities"][:]
            result["protein_proximities"] = torch.zeros(self._protein_maxlen, self._protein_maxlen, 1,
                                                      device=self._device, dtype=torch.float)
            result["protein_proximities"][:prox_data.shape[0], :prox_data.shape[0], :] = torch.tensor(prox_data, device=self._device, dtype=torch.float)

            return result

    @staticmethod
    def collate(data_entries: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:

        keys = set([])
        for e in data_entries:
            keys |= e.keys()

        result = {}
        for key in keys:
            if isinstance(e[key], torch.Tensor) or isinstance(e[key], float) or isinstance(e[key], int):
                result[key] = torch.stack([e[key] for e in data_entries])
            else:
                result[key] = [e[key] for e in data_entries]

        return result

