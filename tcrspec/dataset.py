import os
import sys
import csv
import logging
from typing import Optional, List, Tuple, Dict, Union
from math import log
from enum import Enum

import h5py
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
from .preprocess import PREPROCESS_KD_NAME, PREPROCESS_CLASS_NAME, PREPROCESS_PROTEIN_NAME, PREPROCESS_LOOP_NAME
from .modules.sequence_encoding import mask_loop_left_center_right


_log = logging.getLogger(__name__)


def get_entry_names(hdf5_path: str) -> List[str]:
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        return list(hdf5_file.keys())


class ProteinLoopDataset(Dataset):
    def __init__(self,
                 hdf5_path: str,
                 device: torch.device,
                 loop_maxlen: int,
                 protein_maxlen: int,
                 entry_names: Optional[List[str]] = None,
    ):
        self.name = os.path.splitext(os.path.basename(hdf5_path))[0]

        self._hdf5_path = hdf5_path
        self._device = device
        self._loop_maxlen = loop_maxlen
        self._protein_maxlen = protein_maxlen

        if entry_names is not None:
            self._entry_names = entry_names
        else:
            self._entry_names = get_entry_names(self._hdf5_path)

    @property
    def entry_names(self) -> List[str]:
        return self._entry_names

    def __len__(self) -> int:
        return len(self._entry_names)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:

        entry_name = self._entry_names[index]

        try:
            return self.get_entry(entry_name)
        except Exception as e:
            raise RuntimeError(f"in entry {entry_name}: {str(e)}")

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

            elif PREPROCESS_CLASS_NAME in entry_group:
                result["class"] = torch.tensor(entry_group[PREPROCESS_CLASS_NAME][()], device=self._device, dtype=torch.long)

            for prefix, max_length, start_index in [(PREPROCESS_PROTEIN_NAME, self._protein_maxlen, self._loop_maxlen + 3),
                                                    (PREPROCESS_LOOP_NAME, self._loop_maxlen, 0)]:

                aatype_data = entry_group[prefix]["aatype"][:]
                length = aatype_data.shape[0]
                if length < 3:
                    raise ValueError(f"{entry_name} {prefix} length is {length}")

                elif length > max_length:
                    raise ValueError(f"{entry_name} {prefix} length is {length}, which is larger than the max {max_length}")

                if prefix == "loop":
                    index = mask_loop_left_center_right(length, max_length)
                else:
                    # For the protein, put all residues leftmost
                    index = torch.zeros(max_length, device=self._device, dtype=torch.bool)
                    index[:length] = True

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

                # alphafold needs each connected pair of residues to be one index apart
                result[f"{prefix}_residue_index"] = torch.zeros(max_length, dtype=torch.long, device=self._device)
                result[f"{prefix}_residue_index"][index] = torch.arange(start_index,
                                                                        start_index + length,
                                                                        1, device=self._device)

                result[f"{prefix}_residue_numbers"] = torch.zeros(max_length, dtype=torch.int, device=self._device)
                result[f"{prefix}_residue_numbers"][index] = torch.tensor(entry_group[prefix]["residue_numbers"][:], dtype=torch.int, device=self._device)

                residx_atom14_to_atom37_data = entry_group[prefix]["residx_atom14_to_atom37"][:]
                result[f"{prefix}_residx_atom14_to_atom37"] = torch.zeros((max_length, residx_atom14_to_atom37_data.shape[1]), device=self._device, dtype=torch.long)
                result[f"{prefix}_residx_atom14_to_atom37"][index] = torch.tensor(residx_atom14_to_atom37_data, device=self._device, dtype=torch.long)

                result[f"{prefix}_sequence_onehot"] = torch.zeros((max_length, 32), device=self._device, dtype=torch.float)
                t = torch.tensor(entry_group[prefix]["sequence_onehot"][:], device=self._device, dtype=torch.float)
                result[f"{prefix}_sequence_onehot"][index, :t.shape[1]] = t

                for field_name in ["backbone_rigid_tensor",
                                   "torsion_angles_sin_cos", "alt_torsion_angles_sin_cos",
                                   "atom14_gt_positions", "atom14_alt_gt_positions",
                                   "all_atom_positions",
                                   "torsion_angles_mask", "atom14_gt_exists", "all_atom_mask"]:

                    data = entry_group[prefix][field_name][:]
                    t = torch.zeros([max_length] + list(data.shape[1:]), device=self._device, dtype=torch.float)
                    t[index] = torch.tensor(data, device=self._device, dtype=torch.float)

                    result[f"{prefix}_{field_name}"] = t

            prox_data = entry_group[PREPROCESS_PROTEIN_NAME]["proximities"][:]
            result["protein_proximities"] = torch.zeros(self._protein_maxlen, self._protein_maxlen, 1,
                                                        device=self._device, dtype=torch.float)

            result["protein_proximities"][:prox_data.shape[0], :prox_data.shape[1], :] = torch.tensor(prox_data, device=self._device, dtype=torch.float)

            return result

    @staticmethod
    def collate(data_entries: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:

        keys = None
        for e in data_entries:
            if keys is None:
                keys = e.keys()
            else:
                keys &= e.keys()

        result = {}
        for key in keys:
            if isinstance(e[key], torch.Tensor) or isinstance(e[key], float) or isinstance(e[key], int):
                result[key] = torch.stack([e[key] for e in data_entries])
            else:
                result[key] = [e[key] for e in data_entries]

        return result

