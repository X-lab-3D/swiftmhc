import logging

import torch

from tcrspec.ipa import InvariantPointAttention as IPA
from openfold.utils.rigid_utils import Rigid, Rotation


_log = logging.getLogger(__name__)


def test_ipa():

    n_sequences = 10
    n_residues = 100
    n_sequence_channels = 6
    n_pair_channels = 1
    n_hidden_channels = 16
    n_heads = 8
    n_qk_points = 4
    n_v_points = 6

    module = IPA(n_sequence_channels, n_pair_channels, n_hidden_channels, n_heads, n_qk_points, n_v_points)

    sequences = torch.rand(n_sequences, n_residues, n_sequence_channels)

    proximities = torch.rand(n_sequences, n_residues, n_residues, n_pair_channels)

    rotations = torch.rand(n_sequences, n_residues, 3, 3)
    translations = torch.rand(n_sequences, n_residues, 3)
    transformations = Rigid(Rotation(rot_mats = rotations), translations)

    mask = torch.zeros(n_sequences, n_residues)

    updated_sequences, attention_weights = module(sequences, proximities, transformations, mask)

    assert updated_sequences.shape == sequences.shape, f"{output.shape} != {sequences.shape}"
    assert torch.any(updated_sequences != sequences), "the output equals the input sequences"

    assert attention_weights.shape == (n_sequences, n_residues, n_residues, n_heads), f"attention weights shape is {attention_weights.shape}"
