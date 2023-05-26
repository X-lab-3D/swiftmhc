import sys
from enum import Enum
from typing import Union

import torch
from sklearn.decomposition import PCA


class ComplexClass(Enum):
    NONBINDING = 0
    BINDING = 1

    @staticmethod
    def from_string(s: str):
        if s.upper() == 'NONBINDING':
            return ComplexClass.NONBINDING

        elif s.upper() == 'BINDING':
            return ComplexClass.BINDING

        raise ValueError(s)

    @staticmethod
    def from_int(i: int):
        if i > 0:
            return ComplexClass.BINDING
        else:
            return ComplexClass.NONBINDING

    def __float__(self) -> float:
        return float(self.value)

    def __int__(self) -> int:
        return int(self.value)


class ComplexTableEntry:
    def __init__(self,
                 allele_name: str,
                 epitope_sequence: str,
                 cdr1a_sequence: str,
                 cdr2a_sequence: str,
                 cdr3a_sequence: str,
                 cdr1b_sequence: str,
                 cdr2b_sequence: str,
                 cdr3b_sequence: str,
                 classification: ComplexClass):

        self.allele_name = allele_name

        self.epitope_sequence = epitope_sequence

        self.cdr1a_sequence = cdr1a_sequence
        self.cdr2a_sequence = cdr2a_sequence
        self.cdr3a_sequence = cdr3a_sequence

        self.cdr1b_sequence = cdr1b_sequence
        self.cdr2b_sequence = cdr2b_sequence
        self.cdr3b_sequence = cdr3b_sequence

        self.classification = classification

    def get_cdr_sequence(self):
        return (self.cdr2a_sequence + self.cdr1a_sequence + self.cdr3a_sequence +
                self.cdr3b_sequence + self.cdr2b_sequence + self.cdr1b_sequence)

    def get_model_name(self):
        return self.allele_name.replace('*','x').replace(':', '_') + '-B2M-' + self.epitope_sequence


class StructureDataEntry:
    def __init__(self,
                 embedding: torch.Tensor,
                 mask: torch.Tensor,
                 pocket_pos: torch.Tensor,
                 pocket_dir: torch.Tensor,
                 pairwise: torch.Tensor,
                 translations: torch.Tensor,
                 rotations: torch.Tensor):
        """
        Args:
            embedding: [n_structure_residues, n_sequence_channels]
            mask: [n_structure_residues]
            pocket_pos: [3]
            pocket_dir: [3]
            pairwise: [n_structure_residues, n_structure_residues, n_pairwise_channels]
            translations: [n_structure_residues, 3, 3]
            rotations: [n_structure_residues, 3]
        """

        self.embedding = embedding
        self.mask = mask
        self.pocket_pos = pocket_pos
        self.pocket_dir = pocket_dir
        self.pairwise = pairwise
        self.translations = translations
        self.rotations = rotations

    def to(self, device: torch.device):
        return StructureDataEntry(self.embedding.to(device),
                                  self.mask.to(device),
                                  self.pocket_pos.to(device),
                                  self.pocket_dir.to(device),
                                  self.pairwise.to(device),
                                  self.translations.to(device),
                                  self.rotations.to(device))

    def get_storage_size(self) -> int:

        size = sys.getsizeof(self.embedding.untyped_storage()) + \
               sys.getsizeof(self.mask.untyped_storage()) + \
               sys.getsizeof(self.pocket_pos.untyped_storage()) + \
               sys.getsizeof(self.pairwise.untyped_storage()) + \
               sys.getsizeof(self.translations.untyped_storage()) + \
               sys.getsizeof(self.rotations.untyped_storage())

        return size


class ComplexDataEntry:
    def __init__(self,
                 sequence_embedding: torch.Tensor,
                 structure: StructureDataEntry,
                 target: torch.Tensor):
        """
        Args:
            sequence_embedding: [n_sequence_residues, n_sequence_channels]
            target: 0, 1, or in between
        """

        self.sequence_embedding = sequence_embedding
        self.structure = structure
        self.target = target

    def to(self, device: torch.device):
        return ComplexDataEntry(self.sequence_embedding.to(device),
                                self.structure.to(device),
                                self.target.to(device))

    def get_storage_size(self) -> int:

        size = sys.getsizeof(self.sequence_embedding.untyped_storage()) + \
               self.structure.get_storage_size() + \
               sys.getsizeof(self.target)

        return size
