import torch

import random

from tcrspec.modules.position_encoding import RelativePositionEncoder


def test_positional_encoding():

    batch_size = 10
    maxlen = 16

    encoder = RelativePositionEncoder(2, maxlen, 32)

    l=12

    masks = torch.tensor([[i < l for i in range(maxlen)] for _ in range(batch_size)])

    codes = encoder.relpos(masks)

    for j in range(batch_size):

        code = codes[j]

        assert torch.all(code[0, 0] == code[5, 5])
        assert torch.all(code[0, 1] == code[5, 6])
        assert torch.all(code[1, 0] == code[6, 5])

        assert torch.any(code[0, 10] != code[2, 5])
