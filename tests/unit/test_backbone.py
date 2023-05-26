import logging

import torch

from tcrspec.backbone_update import BackboneUpdate


_log = logging.getLogger(__name__)


def test_backbone_update():

    n_sequences = 10
    n_residues = 100
    n_channels = 6

    module = BackboneUpdate(n_channels)

    sequences = torch.rand(n_sequences, n_residues, n_channels)

    output = module(sequences)

    assert output.shape == (n_sequences, n_residues, 3, 4)
