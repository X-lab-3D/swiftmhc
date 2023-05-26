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

from .models.data import TensorDict
from .models.residue import Residue
from .models.complex import ComplexClass, ComplexTableEntry, ComplexDataEntry, StructureDataEntry
from .domain.amino_acid import amino_acids_by_letter, amino_acids_by_one_hot_index, AMINO_ACID_DIMENSION
from .tools.pdb import get_selected_residues, get_residue_transformations, get_residue_proximities
from .preprocess import PREPROCESS_KD_NAME, PREPROCESS_PROTEIN_NAME, PREPROCESS_LOOP_NAME


_log = logging.getLogger(__name__)


class TargetMode(Enum):
    REGRESSION = 1
    BINARY = 2


class TcrSpecDataset(Dataset):

    PMHC_MAX_LENGTH = 220
    CDR_MAX_LENGTH = 30

    def __init__(self,
                 table_path: str,
                 pmhc_model_directory_path: str,
                 device: torch.device):

        self._table_entries = TcrSpecDataset._get_table_entries(table_path)
        self._pmhc_model_directory_path = pmhc_model_directory_path
        self._device = device

    def __len__(self):
        return len(self._table_entries)

    def __getitem__(self, index: int) -> ComplexDataEntry:
        table_entry = self._table_entries[index]

        pmhc_model_path = os.path.join(self._pmhc_model_directory_path, table_entry.get_model_name() + ".pdb")

        return self._to_data_entry(pmhc_model_path, table_entry.cdr3a_sequence, table_entry.classification)

    def _to_data_entry(self, pmhc_model_path: str, cdr_sequence: str, class_: Optional[int] = None) -> ComplexDataEntry:

        # embed cdr sequence
        cdr_sequence_embedding = torch.stack([amino_acids_by_letter[cdr_sequence[i]].one_hot_code.float() for i in range(len(cdr_sequence))])

        # make sure the length is right, pad with zeros:
        cdr_length = cdr_sequence_embedding.shape[0]
        if cdr_length > TcrSpecDataset.CDR_MAX_LENGTH:
            raise ValueError(f"CDR sequence {cdr_sequence} ({cdr_length}) is too long: the max is {TcrSpecDataset.CDR_MAX_LENGTH}")
        cdr_sequence_embedding = torch.cat((cdr_sequence_embedding,
                                            torch.zeros((TcrSpecDataset.CDR_MAX_LENGTH - cdr_length,
                                                         cdr_sequence_embedding.shape[1]))),
                                           dim=0)

        # load the model
        pdb = pdb2sql(pmhc_model_path)
        try:
            pmhc_residues = get_residues(pdb)
        finally:
            pdb._close()

        # embed pMHC sequence
        pmhc_sequence_embedding = torch.stack([pmhc_residue.amino_acid.one_hot_code.float() for pmhc_residue in pmhc_residues])

        # make sure the length is right, pad with zeros:
        pmhc_length = pmhc_sequence_embedding.shape[0]
        if pmhc_length > TcrSpecDataset.PMHC_MAX_LENGTH:
            raise ValueError(f"pMHC sequence of {pmhc_model_path} is too long: {pmhc_lenth} > {TcrSpecDataset.PMHC_MAX_LENGTH}")
        pmhc_sequence_embedding = torch.cat((pmhc_sequence_embedding,
                                             torch.zeros((TcrSpecDataset.PMHC_MAX_LENGTH - pmhc_length, pmhc_sequence_embedding.shape[1]))),
                                             dim=0)

        # get pMHC structural properties
        pmhc_translations, pmhc_rotations = get_residue_transformations(pmhc_residues)
        pmhc_proximities = get_residue_proximities(pmhc_residues)

        # make sure the length is right, pad with identity transformations:
        pmhc_difference = TcrSpecDataset.PMHC_MAX_LENGTH - pmhc_length

        padding_identity_translations = torch.tensor([0.0, 0.0, 0.0]).repeat(pmhc_difference, 1)

        padding_identity_rotations = torch.tensor([[1.0, 0.0, 0.0],
                                                   [0.0, 1.0, 0.0],
                                                   [0.0, 0.0, 1.0]]).repeat(pmhc_difference, 1, 1)

        pmhc_proximities = torch.cat((torch.cat((pmhc_proximities, torch.zeros(pmhc_length, pmhc_difference)), dim=1),
                                      torch.zeros(pmhc_difference, TcrSpecDataset.PMHC_MAX_LENGTH)), dim=0)

        pmhc_translations = torch.cat((pmhc_translations, padding_identity_translations), dim=0)
        pmhc_rotations = torch.cat((pmhc_rotations, padding_identity_rotations), dim=0)

        return ComplexDataEntry(cdr_sequence_embedding.to(self._device),
                                pmhc_sequence_embedding.to(self._device),
                                pmhc_proximities.to(self._device),
                                pmhc_translations.to(self._device),
                                pmhc_rotations.to(self._device),
                                class_)

    @staticmethod
    def _get_structures(table_entries: List[ComplexTableEntry], model_directory_path: str) -> Dict[str, StructureDataEntry]:

        structure_dict = {}

        for table_entry in table_entries:

            model_name = table_entry.allele_name.replace("*","x").replace(":", "_") + "-" + table_entry.epitope_sequence
            model_path = os.path.join(model_directory_path, model_name + ".pdb")
            mask_path = os.path.join(model_directory_path, model_name + ".mask")

            if model_path in structure_dict:
                continue

            mask_path = model_path.replace(".pdb", ".mask")

            # load the model
            pdb = pdb2sql(model_path)
            try:
                structure_residues = get_residues(pdb)
            finally:
                pdb._close()

            mask = get_masked_residues(mask_path)
            for chain_id, residue_number, three_letter_code in mask:
                residue_id = (chain_id, residue_number, None)

                for residue in structure_residues:
                    if residue.id == residue_id:
                        if residue.amino_acid.three_letter_code != three_letter_code:
                            raise ValueError(f"{residue_id} {structure_residues[residue_id].amino_acid.three_letter_code} != {three_letter_code}")

                        residue.mask = True
                        break
                else:
                    raise ValueError(f"residue not found in {model_path}: {residue_id}")

            structure_length = len(structure_residues)
            if structure_length > PmhcDataset.MHC_MAX_LENGTH:
                raise ValueError(f"sequence from {model_path} is too long: {structure_length}")

            structure_max_difference = PmhcDataset.MHC_MAX_LENGTH - structure_length

            structure_embedding = torch.stack([residue.amino_acid.one_hot_code.float() for residue in structure_residues])
            structure_embedding = torch.cat((structure_embedding, torch.zeros(structure_max_difference, structure_embedding.shape[1])), dim=0)

            structure_mask = torch.tensor([residue.mask for residue in structure_residues])
            structure_mask =  torch.cat((structure_mask, torch.zeros(structure_max_difference)), dim=0)

            # get structural properties
            structure_translations, structure_rotations = get_residue_transformations(structure_residues)
            structure_proximities = get_residue_proximities(structure_residues)

            # pad transformations with identity values
            padding_identity_translations = torch.tensor([0.0, 0.0, 0.0]).repeat(structure_max_difference, 1)

            padding_identity_rotations = torch.tensor([[1.0, 0.0, 0.0],
                                                       [0.0, 1.0, 0.0],
                                                       [0.0, 0.0, 1.0]]).repeat(structure_max_difference, 1, 1)

            structure_translations = torch.cat((structure_translations, padding_identity_translations), dim=0)
            structure_rotations = torch.cat((structure_rotations, padding_identity_rotations), dim=0)

            # pad pairwise proximities with zeros
            structure_proximities = torch.cat((torch.cat((structure_proximities, torch.zeros(structure_length, structure_max_difference)), dim=1),
                                              torch.zeros(structure_max_difference, PmhcDataset.MHC_MAX_LENGTH)), dim=0
                                              ).view(PmhcDataset.MHC_MAX_LENGTH, PmhcDataset.MHC_MAX_LENGTH, 1).float()

            structure_dict[allele_name] = StructureDataEntry(structure_embedding,
                                                             structure_mask,
                                                             structure_proximities,
                                                             structure_translations,
                                                             structure_rotations)

        return structure_dict

    @staticmethod
    def _get_table_entries(table_path: str) -> List[ComplexTableEntry]:

        entries = []
        with open(table_path, 'rt') as f:
            r = csv.reader(f, delimiter='\t')

            header = next(r)

            for row in r:
                record = {header[i]: row[i] for i in range(len(header))}

                mhci_allele_name = record['HMC-I Allele']
                epitope_sequence = record['Epitope']
                cdr1a_sequence = record['CDR1a']
                cdr2a_sequence = record['CDR2a']
                cdr3a_sequence = record['CDR3a']
                cdr1b_sequence = record['CDR1b']
                cdr2b_sequence = record['CDR2b']
                cdr3b_sequence = record['CDR3b']

                class_ = None
                if len(row) == 9:
                    class_ = ComplexClass.from_string(record['classification'])

                entries.append(ComplexTableEntry(mhci_allele_name,
                                                 epitope_sequence,
                                                 cdr1a_sequence, cdr2a_sequence, cdr3a_sequence,
                                                 cdr1b_sequence, cdr2b_sequence, cdr3b_sequence,
                                                 class_))

        return entries

    @staticmethod
    def collate(data_entries: List[ComplexDataEntry]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            input_sequence_embeddings: [n_complexes, n_sequence_residues, n_sequence_channels],
            input_structure_embeddings: [n_complexes, n_structure_residues, n_sequence_channels],
            input_structure_pairwise: [n_complexes, n_structure_residues, n_structure_residues, n_pairwise_channels],
            input_structure_transformations: [n_complexes, n_structure_residues],
            target_values: [n_complexes] or None
        """

        input_sequence_embeddings = []
        input_structure_embeddings = []
        input_structure_pairwise = []
        input_structure_translations = []
        input_structure_rotations = []
        target_values = []
        for data_entry in data_entries:
            input_sequence_embeddings.append(data_entry.sequence_embedding)
            input_structure_embeddings.append(data_entry.structure_embedding)
            input_structure_pairwise.append(data_entry.structure_pairwise)
            input_structure_translations.append(data_entry.structure_translations)
            input_structure_rotations.append(data_entry.structure_rotations)

            if data_entry.classification is not None:
                target_values.append(data_entry.classification.value)

        input_sequence_embeddings = torch.stack(input_sequence_embeddings)
        input_structure_embeddings = torch.stack(input_structure_embeddings)
        input_structure_pairwise = torch.stack(input_structure_pairwise)
        input_structure_transformations = Rigid(Rotation(rot_mats=torch.stack(input_structure_rotations)),
                                                torch.stack(input_structure_translations))

        if len(target_values) < len(data_entries):
            target_values = None
        else:
            target_values = torch.tensor(target_values).to(input_sequence_embeddings.device)

        return (input_sequence_embeddings,
                input_structure_embeddings,
                input_structure_pairwise,
                input_structure_transformations,
                target_values)


def _pad(seq: torch.tensor, dim: int, new_len: int, with_: torch.Tensor):

    len_ = seq.shape[dim]
    n = new_len - len_

    if n == 0:
        return seq

    w = with_.unsqueeze(dim=dim)

    for _ in range(n):
        seq = torch.cat((seq, w), dim=dim)

    return seq


class ProteinLoopDataset:
    def __init__(self, hdf5_path: str, device: torch.device, loop_maxlen: int, protein_maxlen: int):
        self._hdf5_path = hdf5_path
        self._device = device
        self._loop_maxlen = loop_maxlen
        self._protein_maxlen = protein_maxlen

        with h5py.File(self._hdf5_path, 'r') as hdf5_file:
            self._entry_names = list(hdf5_file.keys())

    def __len__(self) -> int:
        return len(self._entry_names)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:

        result = {}

        entry_name = self._entry_names[index]
        with h5py.File(self._hdf5_path, 'r') as hdf5_file:
            entry_group = hdf5_file[entry_name]

            result["ids"] = entry_name

            if PREPROCESS_KD_NAME in entry_group:
                result["kd"] = torch.tensor(entry_group[PREPROCESS_KD_NAME][()], device=self._device, dtype=torch.float)
                result["affinity"] = 1.0 - torch.log(result["kd"]) / log(50000)

            for prefix, max_length in [(PREPROCESS_PROTEIN_NAME, self._protein_maxlen),
                                       (PREPROCESS_LOOP_NAME, self._loop_maxlen)]:

                aatype_data = entry_group[prefix]["aatype"]
                length = aatype_data.shape[0]
                result[f"{prefix}_aatype"] = torch.zeros(max_length, device=self._device, dtype=torch.long)
                result[f"{prefix}_aatype"][:length] = torch.tensor(aatype_data, device=self._device, dtype=torch.long)

                result[f"{prefix}_len_mask"] = torch.zeros(max_length, device=self._device, dtype=torch.bool)
                result[f"{prefix}_len_mask"][:length] = 1

                result[f"{prefix}_residue_index"] = torch.arange(0, max_length, 1, device=self._device, dtype=torch.long)

                residx_atom14_to_atom37_data = entry_group[prefix]["residx_atom14_to_atom37"][:]
                result[f"{prefix}_residx_atom14_to_atom37"] = torch.zeros((max_length, residx_atom14_to_atom37_data.shape[1]), device=self._device, dtype=torch.long)
                result[f"{prefix}_residx_atom14_to_atom37"][:length] = torch.tensor(residx_atom14_to_atom37_data, device=self._device, dtype=torch.long)

                result[f"{prefix}_sequence_embedding"] = torch.zeros((max_length, 32), device=self._device, dtype=torch.float)
                t = torch.tensor(entry_group[prefix]["sequence_embedding"][:], device=self._device, dtype=torch.float)
                result[f"{prefix}_sequence_embedding"][:t.shape[0], :t.shape[1]] = t

                for field_name in ["backbone_rigid_tensor",
                                   "torsion_angles_sin_cos", "alt_torsion_angles_sin_cos", "torsion_angles_mask",
                                   "atom14_gt_exists", "atom14_gt_positions", "atom14_alt_gt_positions",
                                   "all_atom_mask", "all_atom_positions"]:

                    data = entry_group[prefix][field_name][:]
                    length = data.shape[0]
                    t = torch.zeros([max_length] + list(data.shape[1:]), device=self._device, dtype=torch.float)
                    t[:length] = torch.tensor(data, device=self._device, dtype=torch.float)

                    result[f"{prefix}_{field_name}"] = t

            proximity_data = entry_group[PREPROCESS_PROTEIN_NAME]["proximities"][:]
            result["protein_proximities"] = torch.zeros(self._protein_maxlen, self._protein_maxlen, 1,
                                                        device=self._device, dtype=torch.float)
            result["protein_proximities"][:proximity_data.shape[0], :proximity_data.shape[0], :] = torch.tensor(proximity_data, device=self._device, dtype=torch.float)

            proximity_data = entry_group["proximities"][:]
            result["proximities"] = torch.zeros(self._loop_maxlen, self._protein_maxlen, 1,
                                                device=self._device, dtype=torch.float)
            result["proximities"][:proximity_data.shape[0], :proximity_data.shape[1], :] = torch.tensor(proximity_data, device=self._device, dtype=torch.float)

            return result

    @staticmethod
    def collate(data_entries: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:

        keys = set([])
        for e in data_entries:
            keys |= e.keys()

        result = {}
        for key in keys:
            if isinstance(e[key], torch.Tensor):
                result[key] = torch.stack([e[key] for e in data_entries])
            else:
                result[key] = [e[key] for e in data_entries]

        return result


class PmhcDataset:
    MHC_MAX_LENGTH = 220
    PEPTIDE_MAX_LENGTH = 9

    def __init__(self,
                 table_path: str,
                 model_directory_path: str,
                 device: torch.device,
                 target_mode: TargetMode,
                 ids_include: Optional[List[str]] = None):
        """
        Args:
            ids_include: if set, then only those ids will be taken from the table
        """

        self._device = device

        self._table_entries = PmhcDataset._get_entries(table_path, target_mode, ids_include)
        self._structure_entries = PmhcDataset._get_structures(self._table_entries, model_directory_path)

    def __len__(self):
        return len(self._table_entries)

    @staticmethod
    def _get_entries(table_path: str, target_mode: TargetMode,
                     ids_include: Union[List[str], None]) -> List[Tuple[str, str, float]]:

        entries = []
        ids_encountered = set([])

        with open(table_path, 'rt') as f:
            r = csv.reader(f)

            header = next(r)

            allele_index = header.index("allele")
            peptide_index = header.index("peptide")
            measurement_index = header.index("measurement_value")
            id_index = header.index("ID")

            for row in r:
                allele = row[allele_index]
                peptide = row[peptide_index]
                measurement = float(row[measurement_index])
                id_ = row[id_index]

                if target_mode == TargetMode.BINARY:
                    if measurement < 500.0:
                        target = ComplexClass.BINDING
                    else:
                        target = ComplexClass.NONBINDING

                elif target_mode == TargetMode.REGRESSION:
                    target = 1.0 - log(measurement) / log(50000)

                else:
                    raise ValueError(self._target_mode)

                if ids_include is None or id_ in ids_include:
                    entries.append((allele, peptide, target))

                ids_encountered.add(id_)

        if ids_include is not None and len(entries) != len(ids_include):
            for id_ in ids_include:
                if id_ not in ids_encountered:
                    raise ValueError(f"no entry found for {id_}")

            raise ValueError(f"{len(entries)} entries found from list of {len(ids_include)}")

        return entries

    @staticmethod
    def _determine_pocket_direction(residue_selection: List[Tuple[str, int, str]],
                                    structure_residues: List[Residue]) -> Tuple[torch.Tensor, torch.Tensor]:

        positions = []

        n_terminal = None
        c_terminal = None

        for i, (chain_id, residue_number, three_letter_code) in enumerate(residue_selection):
            residue_id = (chain_id, residue_number, None)

            if n_terminal is None or residue_selection[n_terminal][1] > residue_number:
                n_terminal = i

            if c_terminal is None or residue_selection[n_terminal][1] < residue_number:
                c_terminal = i

            for residue in structure_residues:
                if residue.id == residue_id:
                    if residue.amino_acid.three_letter_code != three_letter_code:
                        raise ValueError(f"{residue_id} {structure_residues[residue_id].amino_acid.three_letter_code}" +
                                         f" != {three_letter_code}")

                    positions.append(residue.atoms["CA"].position.tolist())
                    break
            else:
                raise ValueError(f"residue not found in model: {residue_id}")

        pca = PCA(n_components=1)
        pca.fit(positions)
        pocket_pos = torch.tensor(pca.mean_)
        pocket_dir = torch.tensor(pca.components_[0])

        # We know that the N-terminal part of the first helix and the
        # C-terminal part of the second helix form the N-terminal pocket.
        position_n_pocket = 0.5 * (torch.tensor(positions[n_terminal]) + torch.tensor(positions[c_terminal]))
        if pca.transform(position_n_pocket.unsqueeze(dim=0)) > 0.0:
            pocket_dir = -1.0 * pocket_dir  # make sure the N-terminal pocket is on the negative direction

        _log.debug(f"defining pocket at {pocket_pos} direction {pocket_dir}")

        return (pocket_pos, pocket_dir)

    @staticmethod
    def _mask_structure_residues(residue_selection: List[Tuple[str, int, str]],
                                 structure_residues: List[Residue]):

        for chain_id, residue_number, three_letter_code in residue_selection:
            residue_id = (chain_id, residue_number, None)

            for residue in structure_residues:
                if residue.id == residue_id:
                    if residue.amino_acid.three_letter_code != three_letter_code:
                        raise ValueError(f"{residue_id} {structure_residues[residue_id].amino_acid.three_letter_code}" +
                                         f" != {three_letter_code}")

                    residue.mask = True
                    break
            else:
                raise ValueError(f"residue not found in model: {residue_id}")

    @staticmethod
    def _get_structures(table_entries: List[Tuple[str, str, float]],
                        model_directory_path: str) -> Dict[str, StructureDataEntry]:

        structure_dict = {}

        for allele_name, peptide_sequence, complex_class in table_entries:

            if allele_name in structure_dict:
                continue

            _log.debug(f"get structure data for {allele_name}")

            model_name = allele_name.replace("*","x").replace(":", "_")
            model_path = os.path.join(model_directory_path, model_name + ".pdb")
            mask_path = os.path.join(model_directory_path, model_name + ".mask")
            pca_path = os.path.join(model_directory_path, model_name + ".pca")

            # load the model
            pdb = pdb2sql(model_path)
            try:
                structure_residues = get_residues(pdb)
            finally:
                pdb._close()

            # apply mask
            mask_selection = get_selected_residues(mask_path)
            PmhcDataset._mask_structure_residues(mask_selection, structure_residues)

            # define a pocket position and direction in 3D
            pca_selection = get_selected_residues(pca_path)
            pocket_pos, pocket_dir = PmhcDataset._determine_pocket_direction(pca_selection, structure_residues)

            # pad with zeros
            structure_length = len(structure_residues)
            if structure_length > PmhcDataset.MHC_MAX_LENGTH:
                raise ValueError(f"sequence from {model_path} is too long: {structure_length}")

            structure_max_difference = PmhcDataset.MHC_MAX_LENGTH - structure_length

            structure_embedding = torch.stack([residue.amino_acid.one_hot_code.float() for residue in structure_residues])
            structure_embedding = torch.cat((structure_embedding, torch.zeros(structure_max_difference, structure_embedding.shape[1])), dim=0)

            structure_mask = torch.tensor([residue.mask for residue in structure_residues])
            structure_mask =  torch.cat((structure_mask, torch.zeros(structure_max_difference)), dim=0)

            # get structural properties
            structure_translations, structure_rotations = get_residue_transformations(structure_residues)
            structure_proximities = get_residue_proximities(structure_residues)

            # pad transformations with identity values
            padding_identity_translations = torch.tensor([0.0, 0.0, 0.0]).repeat(structure_max_difference, 1)

            padding_identity_rotations = torch.tensor([[1.0, 0.0, 0.0],
                                                       [0.0, 1.0, 0.0],
                                                       [0.0, 0.0, 1.0]]).repeat(structure_max_difference, 1, 1)

            structure_translations = torch.cat((structure_translations, padding_identity_translations), dim=0)
            structure_rotations = torch.cat((structure_rotations, padding_identity_rotations), dim=0)

            # pad pairwise proximities with zeros
            structure_proximities = torch.cat((torch.cat((structure_proximities, torch.zeros(structure_length, structure_max_difference)), dim=1),
                                              torch.zeros(structure_max_difference, PmhcDataset.MHC_MAX_LENGTH)), dim=0
                                              ).view(PmhcDataset.MHC_MAX_LENGTH, PmhcDataset.MHC_MAX_LENGTH, 1).float()

            structure_dict[allele_name] = StructureDataEntry(structure_embedding,
                                                             structure_mask,
                                                             pocket_pos, pocket_dir,
                                                             structure_proximities,
                                                             structure_translations,
                                                             structure_rotations)

        return structure_dict

    def __getitem__(self, index: int) -> ComplexDataEntry:

        allele_name, peptide_sequence, complex_score = self._table_entries[index]

        amino_acids = [amino_acids_by_letter[peptide_sequence[i]] for i in range(len(peptide_sequence))]
        sequence_embedding = torch.stack([amino_acid.one_hot_code.float() for amino_acid in amino_acids])

        structure = self._structure_entries[allele_name]

        if isinstance(complex_score, float):
            target = torch.tensor(complex_score, dtype=torch.float)
        else:
            target = torch.tensor(complex_score, dtype=torch.long)

        return ComplexDataEntry(sequence_embedding, structure, target).to(self._device)

    def get_storage_size(self) -> int:

        size = 0

        for allele_name, peptide_sequence, complex_score in self._table_entries:

            size += sys.getsizeof(allele_name) + sys.getsizeof(peptide_sequence) + sys.getsizeof(complex_score)

        return size

    @staticmethod
    def collate(data_entries: List[ComplexDataEntry]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            input_sequence_embeddings: [n_complexes, n_sequence_residues, n_sequence_channels],
            input_structure_embeddings: [n_complexes, n_structure_residues, n_sequence_channels],
            input_structure_mask: [n_complexes, n_structure_residues],
            input_structure_pairwise: [n_complexes, n_structure_residues, n_structure_residues, n_pairwise_channels],
            input_structure_transformations: [n_complexes, n_structure_residues],
            target_values: [n_complexes] or None
        """

        data_entries_size = sum([data_entry.get_storage_size() for data_entry in data_entries])
        _log.debug(f"collating {data_entries_size} bytes of entries..")

        input_sequence_embeddings = []
        input_structure_embeddings = []
        input_structure_masks = []
        input_pocket_pos = []
        input_pocket_dir = []
        input_structure_pairwise = []
        input_structure_translations = []
        input_structure_rotations = []
        target_values = []
        for data_entry in data_entries:
            input_sequence_embeddings.append(data_entry.sequence_embedding)
            input_structure_embeddings.append(data_entry.structure.embedding)
            input_structure_masks.append(data_entry.structure.mask)
            input_pocket_pos.append(data_entry.structure.pocket_pos)
            input_pocket_dir.append(data_entry.structure.pocket_dir)
            input_structure_pairwise.append(data_entry.structure.pairwise)
            input_structure_translations.append(data_entry.structure.translations)
            input_structure_rotations.append(data_entry.structure.rotations)

            if data_entry.target is not None:
                target_values.append(data_entry.target)

        input_sequence_embeddings = torch.stack(input_sequence_embeddings)
        input_structure_embeddings = torch.stack(input_structure_embeddings)
        input_structure_masks = torch.stack(input_structure_masks)
        input_pocket_pos = torch.stack(input_pocket_pos)
        input_pocket_dir = torch.stack(input_pocket_dir)
        input_structure_pairwise = torch.stack(input_structure_pairwise)
        input_structure_rotations = torch.stack(input_structure_rotations)
        input_structure_translations = torch.stack(input_structure_translations)

        if len(target_values) < len(data_entries):
            target_values = None
        else:
            target_values = torch.stack(target_values)

        return (input_sequence_embeddings,
                input_structure_embeddings,
                input_structure_masks,
                input_pocket_pos,
                input_pocket_dir,
                input_structure_pairwise,
                input_structure_translations,
                input_structure_rotations,
                target_values)

