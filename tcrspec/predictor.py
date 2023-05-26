import csv
import sys
from typing import List, Optional, Union, Tuple
import logging
from uuid import uuid4
from math import ceil
import random
from time import time

import torch
import torch.nn
from torch.nn.modules.transformer import TransformerEncoderLayer

from openfold.utils.rigid_utils import Rigid, Rotation

from .ipa import InvariantPointAttention as IPA
from .backbone_update import BackboneUpdate
from .position_encoding import PositionalEncoding
from .domain.amino_acid import AMINO_ACID_DIMENSION, amino_acids_by_letter, amino_acids_by_one_hot_index, unknown_amino_acid
from .dataset import TargetMode
from .models.complex import ComplexClass
from .models.amino_acid import AminoAcid
from .utils import zero_pad, one_hot_decode_sequence


_log = logging.getLogger(__name__)


def _amino_acid_table_value(amino_acid: Optional[AminoAcid]):

    if amino_acid is None:
        return "<gap>"

    else:
        return amino_acid.name


def _matrix_to_table(path: str,
                     top_labels: List[str], left_labels: List[str],
                     matrix: torch.Tensor):
    """
    Args:
        path: where to store the csv table
        top_labels: [n_columns], used as header
        left_labels: [n_rows], used as header
        matrix: [n_rows, n_columns], actual numbers to be placed in the table
    """

    n_rows, n_columns = matrix.shape

    with open(path, 'wt') as file_:
        w = csv.writer(file_)

        w.writerow(["\\"] + top_labels)

        for row_index in range(n_rows):

            w.writerow([left_labels[row_index]] + matrix[row_index].tolist())


def _linear_embeddings_to_tables(prefix: str,
                                 input_sequences: torch.Tensor,
                                 embeddings: torch.Tensor):
    """
    Args:
        input_sequences: [n_complexes, n_residues, AMINO_ACID_DIMENSION]
        embeddings: [n_complexes, n_head, n_residues, embed_dimension]
    """

    n_complexes, n_head, n_residues, embed_dimension = embeddings.shape

    complex_index_range = list(range(n_complexes))

    # select a random subset of complexes
    selected_complex_indices = [random.choice(complex_index_range) for _ in range(1)]

    for complex_index in selected_complex_indices:

        sequence = one_hot_decode_sequence(input_sequences[complex_index])

        complex_id = uuid4().hex

        sequence_name = "".join([amino_acid.one_letter_code for amino_acid in sequence if amino_acid is not None])

        file_path = prefix

        # Short sequences (like loops) fit in the file name.
        if len(sequence_name) < 20:
            file_path += "-" + sequence_name

        file_path += ".csv"

        top_labels = [f"{index + 1} " + _amino_acid_table_value(sequence[index])
                      for index in range(n_residues)]

        left_labels = [f"dim {index}" for index in range(embed_dimension)]

        # reduce the heads dimension by taking the mean
        matrix = torch.mean(embeddings[complex_index], dim=0).transpose(0, 1)

        _matrix_to_table(file_path, top_labels, left_labels, matrix)


def _residue_weights_to_tables(prefix: str,
                               input_sequences_dim1: torch.Tensor,
                               input_sequences_dim2: torch.Tensor,
                               attention_weights: torch.Tensor):

    """
    Args:
        input_sequences_dim1: [n_complexes, n_dim1_residues, AMINO_ACID_DIMENSION]
        input_sequences_dim2: [n_complexes, n_dim2_residues, AMINO_ACID_DIMENSION]
        attention_weights: [n_complexes, n_dim1_residues, n_dim2_residues, n_head]
    """

    n_complexes, n_dim1_residues, n_dim2_residues, n_head = attention_weights.shape

    complex_index_range = list(range(n_complexes))

    # select a random subset of complexes
    selected_complex_indices = [random.choice(complex_index_range) for _ in range(1)]

    for complex_index in selected_complex_indices:

        sequence_dim1 = one_hot_decode_sequence(input_sequences_dim1[complex_index])
        sequence_dim2 = one_hot_decode_sequence(input_sequences_dim2[complex_index])

        complex_id = uuid4().hex

        sequence_name = "".join([amino_acid.one_letter_code for amino_acid in sequence_dim1 if amino_acid is not None])

        file_path = prefix

        # Short sequences (like loops) fit in the file name.
        if len(sequence_name) < 20:
            file_path += "-" + sequence_name

        file_path += ".csv"

        top_labels = [f"{index_dim1 + 1} " + _amino_acid_table_value(sequence_dim1[index_dim1])
                      for index_dim1 in range(len(sequence_dim1))]

        left_labels = [f"{index_dim2 + 1} " + _amino_acid_table_value(sequence_dim2[index_dim2])
                       for index_dim2 in range(len(sequence_dim2))]

        # reduce the heads dimension by taking the mean
        matrix = torch.mean(attention_weights[complex_index], dim=2).transpose(0, 1)

        _matrix_to_table(file_path, top_labels, left_labels, matrix)


class Predictor(torch.nn.Module):
    def  __init__(self, run_id: str, struct_maxlen: int, loop_maxlen: int, n_head: int,
                  target_mode: TargetMode):

        super(Predictor, self).__init__()

        self._run_id = run_id
        self._target_mode = target_mode

        n_sequence_channels = AMINO_ACID_DIMENSION
        loop_len = loop_maxlen
        struct_len = struct_maxlen
        embd_dim = int(ceil(n_sequence_channels / n_head)) * n_head
        n_pair_channels = 1
        n_hidden_channels = 16
        n_query_points = 4
        n_point_values = 8

        self._loop_norm = torch.nn.Sequential(
            torch.nn.LayerNorm((loop_len, n_sequence_channels)),
            torch.nn.Dropout(p=0.1)
        )

        self._struct_norm = torch.nn.Sequential(
            torch.nn.LayerNorm((struct_len, n_sequence_channels)),
            torch.nn.Dropout(p=0.1)
        )

        self._pair_norm = torch.nn.Sequential(
            torch.nn.LayerNorm((struct_len, struct_len, n_pair_channels)),
            torch.nn.Dropout(p=0.1)
        )

        self._loop_q_gen = torch.nn.Linear(embd_dim * loop_len, loop_len * embd_dim * n_head)

        self._struct_k_gen = torch.nn.Sequential(
            torch.nn.Linear(n_sequence_channels * struct_len, struct_len * embd_dim * n_head),
            torch.nn.LayerNorm(struct_len * embd_dim * n_head)
        )

        self._ipa1 = IPA(n_sequence_channels, n_pair_channels, n_hidden_channels, n_head, n_query_points, n_point_values)

        self._pos_enc = PositionalEncoding(embd_dim, 9)
        self._transf_enc = TransformerEncoderLayer(embd_dim, n_head, batch_first=True, dropout=0.1)

        self._n_head = n_head
        self._embd_dim = embd_dim
        self._sqrt_embd_dim = torch.sqrt(torch.tensor(embd_dim, dtype=float))

        self._loop_pos3d_gen = torch.nn.Linear(3, 3)

        if target_mode == TargetMode.REGRESSION:
            self._result = torch.nn.Sequential(
                torch.nn.Linear(loop_len * struct_len * n_head, 1)
            )
        elif target_mode == TargetMode.BINARY:
            self._result = torch.nn.Sequential(
                torch.nn.Linear(loop_len * struct_len * n_head, 2),
                torch.nn.Softmax(dim=1)
            )
        else:
            raise ValueError(target_mode)

    def save_ipa(self, path: str):
        torch.save(self._ipa1.state_dict(), path)

    def load_ipa(self, path: str, device: torch.device):
        self._ipa1.load_state_dict(torch.load(path, map_location=device))
        self._ipa1.eval()
        self._ipa1.requires_grad_(False)

    def get_storage_size(self) -> int:
        size = 0
        for parameter in self.parameters():
            size += sys.getsizeof(parameter.untyped_storage())
        return size

    def forward(self,
                in_loop_embd: torch.Tensor,
                in_struct_embd: torch.Tensor,
                in_struct_mask: torch.Tensor,
                in_struct_pair: torch.Tensor,
                in_struct_transl: torch.Tensor,
                in_struct_rot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            in_loop_embd: [batch_length, n_loop_residues, n_sequence_channels]
            in_struct_embd: [batch_length, n_structure_residues, n_sequence_channels]
            in_struct_mask: [batch_length, n_structure_residues]
            in_struct_pair: [batch_length, n_structure_residues, n_structure_residues, n_pairwise_channels]
            in_struct_transl: [batch_length, n_structure_residues, 3]
            in_struct_rot: [batch_length, n_structure_residues, 3, 3]
        Returns:
            [1] predicted probability: [n_complexes] for regression or [n_complexes, 2] for binary target mode
            [2] predicted loop positions: [n_complexes, n_loop_residues, 3]
            [3] cross attention weights: [n_compleses, n_head, n_loop_residues, n_structure_residues]
        """

        batch_len = in_loop_embd.shape[0]
        loop_len = in_loop_embd.shape[1]
        seq_dim = in_loop_embd.shape[2]
        struct_len = in_struct_embd.shape[1]

        # [n_complexes, n_structure_residues]
        ipa_mask = torch.ones(batch_len, struct_len).to(in_struct_embd.device)

        # [n_complexes, n_structure_residues, n_sequence_channels]
        struct_embd = self._struct_norm(in_struct_embd)

        # [n_complexes, n_structure_residues, n_structure_residues, n_pairwise_channels]
        struct_pair = self._pair_norm(in_struct_pair)

        # [n_complexes, n_structure_residues, n_sequence_channels], [n_complexes, n_structure_residues, n_structure_residues, n_head]
        struct_embd, ipa_att_w = self._ipa1(struct_embd,
                                            struct_pair,
                                            Rigid(Rotation(rot_mats=in_struct_rot), in_struct_transl),
                                            ipa_mask)

        # [n_complexes, n_loop_residues, input_dimension]
        loop_embd = self._loop_norm(in_loop_embd)

        # [n_complexes, n_loop_residues, embed_dimension]
        loop_embd = zero_pad(loop_embd, 2, self._embd_dim)

        # [n_complexes, n_loop_residues, embed_dimension]
        loop_embd = self._pos_enc(loop_embd)

        # [n_complexes, n_loop_residues, embed_dimension]
        loop_embd = self._transf_enc(loop_embd)

        # [n_complexes, n_head, n_loop_residues, embed_dimension]
        loop_q = self._loop_q_gen(
            loop_embd.view(batch_len, loop_len * self._embd_dim)
        ).view(batch_len, loop_len, self._n_head, self._embd_dim).permute(0, 2, 1, 3)

        # [n_complexes, n_head, n_structure_residues, embed_dimension]
        struct_k = self._struct_k_gen(
            struct_embd.view(batch_len, struct_len * seq_dim)
        ).view(batch_len, struct_len, self._n_head, self._embd_dim).permute(0, 2, 1, 3)

        # [n_complexes, n_head, n_loop_residues, n_structure_residues]
        cross_att_w = torch.matmul(loop_q, struct_k.transpose(2, 3)) / self._sqrt_embd_dim

        # apply mask -> [n_complexes, n_head, n_loop_residues, n_structure_residues]
        cross_att_w *= in_struct_mask.unsqueeze(dim=1).unsqueeze(dim=2).expand(-1, self._n_head, loop_len, -1)

        # [n_complexes, n_head, n_loop_residues, n_structure_residues]
        cross_att_w = torch.softmax(cross_att_w, dim=3)

        # Multiply structure positions by their attention weights, of which the sum is 1.0.
        # The sum of these multiplied vectors will be the predicted loop positions.
        # Divide by the number of heads to normalize.
        # output dimensions: [n_complexes, n_loop_residues, 3]
        loop_att_pos_sums = torch.sum(cross_att_w.sum(dim=1).unsqueeze(dim=3)
                                      * in_struct_transl.unsqueeze(dim=1), dim=2) / self._n_head

        # Predict the actual positions using a linear layer.
        loop_positions = self._loop_pos3d_gen(loop_att_pos_sums)

        # [n_complexes, n_outputs]
        p = self._result(torch.reshape(cross_att_w, (batch_len, self._n_head * loop_len * struct_len)))

        if self._target_mode == TargetMode.REGRESSION:
            p = p.view(p.shape[0])

        return (p, loop_positions, cross_att_w)
