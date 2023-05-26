import torch

from tcrspec.modules.position_encoding import PositionalEncoding, get_onehot_positions


def test_positional_encoding():
    length = 16
    depth = 64
    batch_size = 32

    encoder = PositionalEncoding(depth, length)

    z = torch.zeros(batch_size, length, depth)

    encoded = encoder(z)

    for batch_index in range(batch_size):

        for pos in range(1, length):

            assert torch.any(encoded[batch_index, pos - 1] != encoded[batch_index, pos]), f"at batch {batch_index}: position encoded {pos - 1} same as {pos}"
