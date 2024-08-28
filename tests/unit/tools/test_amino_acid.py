import torch

from swiftmhc.tools.amino_acid import stretch_sequence


def test_stretch_sequence():
    s = torch.nn.functional.one_hot(torch.tensor([0, 1, 7, 18, 12, 3, 5, 9, 4]), num_classes=32)

    ss, m = stretch_sequence(s, 16)

    assert torch.all(ss[m, :] == s)

    assert m.sum().item() == s.shape[0]
