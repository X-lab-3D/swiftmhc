from math import pi, ceil, floor, log
import logging

import torch
import torch.nn
from torch.nn.functional import one_hot


_log = logging.getLogger(__name__)


def get_relative_positions(s: torch.Tensor) -> torch.Tensor:
    """
    Args:
        s: [n_sequences, max_length, ?]
    Returns:
        [n_sequences, max_length] ranging (0 - 1)
    """

    # [n_sequences, max_length] (booleans)
    sequences_not_gap = torch.any(s != 0.0, dim=2)

    # [n_sequences] (integers)
    sequences_lengths = torch.count_nonzero(sequences_not_gap, dim=1)

    # [n_sequences, max_length]
    relative_positions = torch.zeros((s.shape[0], s.shape[1]))
    for sequence_index in range(s.shape[0]):

        step_size = 1.0 / (sequences_lengths[sequence_index] - 1.0)

        relative_positions[sequence_index, sequences_not_gap[sequence_index]] = torch.arange(sequences_lengths[sequence_index]) * step_size

    return relative_positions


def get_onehot_positions(s: torch.Tensor) -> torch.Tensor:
    """
    Args:
        s: [n_sequences, max_length, ?]
    Returns:
        [n_sequences, max_length, max_length]
    """

    encodings = torch.zeros(s.shape[0], s.shape[1], s.shape[1], dtype=float)

    for index in range(s.shape[1]):
        encodings[:,index, index] = 1.0

    return encodings


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, max_len):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe
        return x

