
import torch

from swiftmhc.tools.md import build_modeller


def test_modeller():

        modeller = build_modeller(
            [
                ('A', torch.tensor([0, 0]), torch.rand(2, 14), torch.tensor([[True] * 5 + [False] * 9] * 2)),
            ]
        )
