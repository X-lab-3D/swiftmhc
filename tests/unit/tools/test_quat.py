from math import sqrt

import torch

from swiftmhc.tools.quat import rotate_vec_by_quat


epsilon = 1e-6


def test_no_rotation():

    q = torch.tensor([1.0, 0.0, 0.0, 0.0])
    v = torch.tensor([1.0, 0.0, 0.0])

    r = rotate_vec_by_quat(q, v)

    assert ((r - v).abs() < epsilon).all(), f"{r} != {v}"


def test_x_to_y():

    a = sqrt(0.5)

    q = torch.tensor([a, 0.0, 0.0, a])  # 90 degrees around z-axis
    v = torch.tensor([1.0, 0.0, 0.0])

    r = rotate_vec_by_quat(q, v)

    e = torch.tensor([0.0, 1.0, 0.0])

    assert ((r - e).abs() < epsilon).all(), f"{r} != {e}"


def test_x_flip():

    q = torch.tensor([0.0, 0.0, 0.0, 1.0])  # 180 degrees around z-axis
    v = torch.tensor([1.0, 0.0, 0.0])

    r = rotate_vec_by_quat(q, v)

    e = torch.tensor([-1.0, 0.0, 0.0])

    assert ((r - e).abs() < epsilon).all(), f"{r} != {e}"
