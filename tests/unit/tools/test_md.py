
import torch

from swiftmhc.tools.md import build_modeller


def test_modeller():

        modeller = build_modeller(
            [
                ('A', torch.tensor([1, 2]), torch.tensor([0, 0]), torch.rand(2, 14, 3), torch.tensor([[True] * 5 + [False] * 9] * 2)),
            ]
        )

        modeller.addHydrogens()
