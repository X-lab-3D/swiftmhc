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

from openfold.data.data_transforms import make_atom14_masks
from openfold.utils.rigid_utils import Rigid
from openfold.np import residue_constants

from blosum import BLOSUM

from .models.types import ModelType
from .models.data import TensorDict
from .models.residue import Residue
from .domain.amino_acid import amino_acids_by_letter, amino_acids_by_one_hot_index, AMINO_ACID_DIMENSION, canonical_amino_acids
from .tools.pdb import get_selected_residues, get_residue_transformations, get_residue_proximities
from .preprocess import (
    PREPROCESS_AFFINITY_NAME,
    PREPROCESS_AFFINITY_LT_MASK_NAME,
    PREPROCESS_AFFINITY_GT_MASK_NAME,
    PREPROCESS_CLASS_NAME,
    PREPROCESS_PROTEIN_NAME,
    PREPROCESS_PEPTIDE_NAME,
    affinity_binding_threshold,
)

blosum62_matrix = BLOSUM(62)
blosum62_depth = len(canonical_amino_acids)
blosum62_codes = {aa1: [blosum62_matrix[aa1.one_letter_code][aa2.one_letter_code]
                        for aa2 in canonical_amino_acids]
                  for aa1 in canonical_amino_acids}


_log = logging.getLogger(__name__)


def get_entry_names(
    hdf5_path: str,
) -> List[str]:
    """
    Get the names of the entry groups directly under the HDF5 data file

    Args:
        hdf5_path: data file to get the entry names from
    Returns:
        the entry names in the data file
    """

    with h5py.File(hdf5_path, 'r') as hdf5_file:
        return list(hdf5_file.keys())


class ProteinLoopDataset(Dataset):
    def __init__(
        self,
        hdf5_path: str,
        device: torch.device,
        peptide_maxlen: int,
        protein_maxlen: int,
        entry_names: Optional[List[str]] = None,
        pairs: Optional[List[Tuple[str, str]]] = None,
    ):
        """
        Agrs:
            hdf5_path: hdf5 file with structural data and optionally binding affinity
            device: cpu or cuda, must match with model
            peptide_maxlen: maximum length for storage of peptide data (in amino acids)
            protein_maxlen: maximum length for storage of protein data (in amino acids)
            entry_names: optional list of entries to use, by default all entries in the hdf5 are used
            pairs: optional list of pairs (peptide and allele) to combine, used for predicting unlabeled data with no structure
        """

        self.name = os.path.splitext(os.path.basename(hdf5_path))[0]

        self._hdf5_path = hdf5_path
        self._device = device
        self._peptide_maxlen = peptide_maxlen
        self._protein_maxlen = protein_maxlen

        if entry_names is not None:
            self._entry_names = entry_names
        else:
            self._entry_names = get_entry_names(self._hdf5_path)

        self._pairs = pairs

    @property
    def entry_names(self) -> List[str]:
        return self._entry_names

    def __len__(self) -> int:
        if self._pairs is not None:
            return len(self._pairs)
        else:
            return len(self._entry_names)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:

        if self._pairs is not None:

            # must make the combination
            peptide, allele = self._pairs[index]

            try:
                result = self._get_structural_data(allele, False)
                result.update(self._get_sequence_data(peptide))
                result["peptide"] = peptide
                result["allele"] = allele
                result["ids"] = f"{allele}-{peptide}"

                return result
            except Exception as e:
                raise RuntimeError(f"for pair {peptide},{allele}: {str(e)}")
        else:
            # preprocessed combination from hdf5

            entry_name = self._entry_names[index]

            try:
                return self.get_entry(entry_name)
            except Exception as e:
                raise RuntimeError(f"in entry {entry_name}: {str(e)}")

    def has_entry(self,  entry_name: str) -> bool:
        return entry_name in self._entry_names

    def _get_peptide_sequence(self, entry_name: str) -> str:
        """
        gets the peptide sequence from the hdf5 file, under the given entry name
        """

        with h5py.File(self._hdf5_path, 'r') as hdf5_file:
            entry = hdf5_file[entry_name]
            peptide = entry[PREPROCESS_PEPTIDE_NAME]
            aatype = peptide["aatype"]
            sequence = [residue_constants.restypes[i] for i in aatype]

            return sequence

    def _get_sequence_data(self, sequence: str):

        result = {}

        max_length = self._peptide_maxlen
        prefix = PREPROCESS_PEPTIDE_NAME
        length = len(sequence)
        amino_acids = [amino_acids_by_letter[a] for a in sequence]

        # amino acid mask
        mask = torch.zeros(self._peptide_maxlen, device=self._device, dtype=torch.bool)
        mask[:length] = True
        for interfix in ["self", "cross"]:
            result[f"{prefix}_{interfix}_residues_mask"] = mask

        # amino acid numbers
        result[f"{prefix}_aatype"] = torch.zeros(self._peptide_maxlen, device=self._device, dtype=torch.long)
        result[f"{prefix}_aatype"][:length] = torch.tensor([aa.index for aa in amino_acids], device=self._device)

        # one-hot encoding
        result[f"{prefix}_sequence_onehot"] = torch.zeros((max_length, 32), device=self._device, dtype=torch.float)
        result[f"{prefix}_sequence_onehot"][:length, :AMINO_ACID_DIMENSION] = torch.stack([aa.one_hot_code for aa in amino_acids]).to(device=self._device)
        # blosum62 encoding
        result[f"{prefix}_blosum62"] = torch.zeros((max_length, 32), device=self._device, dtype=torch.float)
        result[f"{prefix}_blosum62"][:length, :blosum62_depth] = torch.tensor([blosum62_codes[aa] for aa in amino_acids],
                                                                            device=self._device)

        # openfold needs each connected pair of residues to be one index apart
        result[f"{prefix}_residue_index"] = torch.zeros(max_length, dtype=torch.long, device=self._device)
        result[f"{prefix}_residue_index"][:length] = torch.arange(0, length, 1, device=self._device)

        # residue numbers
        result[f"{prefix}_residue_numbers"] = torch.zeros(max_length, dtype=torch.long, device=self._device)
        result[f"{prefix}_residue_numbers"][:length] = torch.arange(1, length + 1, 1, device=self._device)

        # atoms mask
        result[f"{prefix}_atom14_gt_exists"] = torch.zeros((max_length, 14), dtype=torch.float, device=self._device)
        result[f"{prefix}_torsion_angles_mask"] = torch.zeros((max_length, 7), dtype=torch.float, device=self._device)
        result[f"{prefix}_all_atom_mask"] = torch.zeros((max_length, 37), dtype=torch.float, device=self._device)
        for i, amino_acid in enumerate(amino_acids):

            result[f"{prefix}_torsion_angles_mask"][i, :3] = 1.0
            for k, mask in enumerate(residue_constants.chi_angles_mask[amino_acid.index]):
                result[f"{prefix}_torsion_angles_mask"][i, 3 + k] = mask

            result[f"{prefix}_atom14_gt_exists"][i] = torch.tensor(residue_constants.restype_atom14_mask[amino_acid.index], device=self._device)
            result[f"{prefix}_all_atom_mask"][i] = torch.tensor(residue_constants.restype_atom37_mask[amino_acid.index], device=self._device)

        for key, value in make_atom14_masks({"aatype": result[f"{prefix}_aatype"]}).items():
            result[f"{prefix}_{key}"] = value

        return result

    def _get_structural_data(self, entry_name: str, take_peptide: bool) -> Dict[str, torch.Tensor]:

        result = {}

        with h5py.File(self._hdf5_path, 'r') as hdf5_file:
            entry_group = hdf5_file[entry_name]

            # Decide whether we take the peptide (if present) or just the protein residue data
            # In this iteration list:
            # 1. the group name: protein or peptide
            # 2. the maximum number of residues to fit, thus how much space to allocate
            # 3. the starting number in the list of indexes
            residue_iteration = [(PREPROCESS_PROTEIN_NAME, self._protein_maxlen, self._peptide_maxlen + 3)]
            if take_peptide:
                residue_iteration.append((PREPROCESS_PEPTIDE_NAME, self._peptide_maxlen, 0))

            for prefix, max_length, start_index in residue_iteration:

                # aatype, needed by openfold
                aatype_data = entry_group[prefix]["aatype"][:]
                length = aatype_data.shape[0]
                if length < 3:
                    raise ValueError(f"{entry_name} {prefix} length is {length}")

                elif length > max_length:
                    raise ValueError(f"{entry_name} {prefix} length is {length}, which is larger than the max {max_length}")

                # Put all residues leftmost
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

                # openfold needs each connected pair of residues to be one index apart
                result[f"{prefix}_residue_index"] = torch.zeros(max_length, dtype=torch.long, device=self._device)
                result[f"{prefix}_residue_index"][index] = torch.arange(start_index,
                                                                        start_index + length,
                                                                        1, device=self._device)

                # get the rest of the data from the HDF5 file
                result[f"{prefix}_residue_numbers"] = torch.zeros(max_length, dtype=torch.int, device=self._device)
                result[f"{prefix}_residue_numbers"][index] = torch.tensor(entry_group[prefix]["residue_numbers"][:], dtype=torch.int, device=self._device)

                residx_atom14_to_atom37_data = entry_group[prefix]["residx_atom14_to_atom37"][:]
                result[f"{prefix}_residx_atom14_to_atom37"] = torch.zeros((max_length, residx_atom14_to_atom37_data.shape[1]), device=self._device, dtype=torch.long)
                result[f"{prefix}_residx_atom14_to_atom37"][index] = torch.tensor(residx_atom14_to_atom37_data, device=self._device, dtype=torch.long)

                result[f"{prefix}_sequence_onehot"] = torch.zeros((max_length, 32), device=self._device, dtype=torch.float)
                t = torch.tensor(entry_group[prefix]["sequence_onehot"][:], device=self._device, dtype=torch.float)
                result[f"{prefix}_sequence_onehot"][index, :t.shape[1]] = t

                result[f"{prefix}_blosum62"] = torch.zeros((max_length, 32), device=self._device, dtype=torch.float)
                t = torch.tensor(entry_group[prefix]["blosum62"][:], device=self._device, dtype=torch.float)
                result[f"{prefix}_blosum62"][index, :t.shape[1]] = t

                for field_name in ["backbone_rigid_tensor",
                                   "torsion_angles_sin_cos", "alt_torsion_angles_sin_cos",
                                   "atom14_gt_positions", "atom14_alt_gt_positions",
                                   "all_atom_positions",
                                   "torsion_angles_mask", "atom14_gt_exists", "all_atom_mask"]:

                    data = entry_group[prefix][field_name][:]
                    t = torch.zeros([max_length] + list(data.shape[1:]), device=self._device, dtype=torch.float)
                    t[index] = torch.tensor(data, device=self._device, dtype=torch.float)

                    result[f"{prefix}_{field_name}"] = t

            # protein residue-residue proximity data
            prox_data = entry_group[PREPROCESS_PROTEIN_NAME]["proximities"][:]
            result["protein_proximities"] = torch.zeros(self._protein_maxlen, self._protein_maxlen, 1,
                                                        device=self._device, dtype=torch.float)

            result["protein_proximities"][:prox_data.shape[0], :prox_data.shape[1], :] = torch.tensor(prox_data, device=self._device, dtype=torch.float)

        return result

    def _set_zero_peptide_structure(self, result: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        all frames, atom positions and angles set to zero
        """

        max_length = self._peptide_maxlen

        result["peptide_backbone_rigid_tensor"] = torch.zeros((max_length, 4, 4), device=self._device, dtype=torch.float)

        result["peptide_torsion_angles_sin_cos"] = torch.zeros((max_length, 7, 2), device=self._device, dtype=torch.float)
        result["peptide_alt_torsion_angles_sin_cos"] = torch.zeros((max_length, 7, 2), device=self._device, dtype=torch.float)

        result["peptide_atom14_gt_positions"] = torch.zeros((max_length, 14, 3), device=self._device, dtype=torch.float)
        result["peptide_atom14_alt_gt_positions"] = torch.zeros((max_length, 14, 3), device=self._device, dtype=torch.float)

        result["peptide_all_atom_positions"] = torch.zeros((max_length, 37, 3), device=self._device, dtype=torch.float)

        return result

    def get_entry(self, entry_name: str) -> Dict[str, torch.Tensor]:
        """
        Gets the data entry (case) with the given name(ID)
        """

        result = {}

        with h5py.File(self._hdf5_path, 'r') as hdf5_file:
            entry_group = hdf5_file[entry_name]

            # store the id with the data entry
            result["ids"] = entry_name

            # The target affinity value is optional, thus only take it if present
            if PREPROCESS_AFFINITY_NAME in entry_group:
                result["affinity"] = torch.tensor(entry_group[PREPROCESS_AFFINITY_NAME][()], device=self._device, dtype=torch.float)

                if PREPROCESS_AFFINITY_LT_MASK_NAME in entry_group:
                    result["affinity_lt"] = torch.tensor(entry_group[PREPROCESS_AFFINITY_LT_MASK_NAME][()], device=self._device, dtype=torch.bool)
                else:
                    result["affinity_lt"] = torch.tensor(False, device=self._device, dtype=torch.bool)

                if PREPROCESS_AFFINITY_GT_MASK_NAME in entry_group:
                    result["affinity_gt"] = torch.tensor(entry_group[PREPROCESS_AFFINITY_GT_MASK_NAME][()], device=self._device, dtype=torch.bool)
                else:
                    result["affinity_gt"] = torch.tensor(False, device=self._device, dtype=torch.bool)

                result["class"] = (result["affinity"] > affinity_binding_threshold)

            if PREPROCESS_CLASS_NAME in entry_group:
                result["class"] = torch.tensor(entry_group[PREPROCESS_CLASS_NAME][()], device=self._device, dtype=torch.long)

        if "class" in result and result["class"] > 0:
            result.update(self._get_structural_data(entry_name, True))
        else:
            # nonbinders need no structural truth data for the peptide
            # only structural data for the protein
            result.update(self._get_structural_data(entry_name, False))

            peptide_sequence = self._get_peptide_sequence(entry_name)
            result.update(self._get_sequence_data(peptide_sequence))
            result = self._set_zero_peptide_structure(result)

        return result

    @staticmethod
    def collate(data_entries: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collation function, to pack data of multiple entries in one batch.
        """

        keys = None
        for e in data_entries:
            if keys is None:
                keys = e.keys()
            else:
                keys &= e.keys()

        result = {}
        for key in keys:
            try:
                if isinstance(e[key], torch.Tensor) or isinstance(e[key], float) or isinstance(e[key], int):
                    result[key] = torch.stack([e[key] for e in data_entries])
                else:
                    result[key] = [e[key] for e in data_entries]

            except RuntimeError as e:
                raise RuntimeError(f"on {key}: {str(e)}")

        return result

