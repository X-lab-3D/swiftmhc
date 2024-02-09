import logging
from typing import List
from math import sin, cos

import torch

from openfold.utils.rigid_utils import Rigid, Rotation

from swiftmhc.operate import average_rigid


_log = logging.getLogger(__name__)


def _get_rotx_mat(a: float) -> List[List[float]]:
    return [[1.0, 0.0, 0.0],
            [0.0, cos(a), -sin(a)],
            [0.0, sin(a), cos(a)]]


def test_average_rigid():

    batch_size = 8
    count = 10
    points = 100 * torch.rand(batch_size, 3)
    translations = torch.rand(batch_size, count, 3)
    angles = torch.rand(batch_size, count)
    rotx_mats = torch.tensor([[_get_rotx_mat(angles[i, j].item()) for j in range(count)] for i in range(batch_size)])

    rigids = Rigid(Rotation(rot_mats=rotx_mats), translations)

    avg = average_rigid(rigids, dim=1)

    mean_translation = torch.mean(translations, dim=1)
    mean_angles = torch.mean(angles, dim=1)
    mean_rotx_mat = torch.tensor([_get_rotx_mat(mean_angles[i]) for i in range(batch_size)])
    expected_results = Rigid(Rotation(rot_mats=mean_rotx_mat), mean_translation).apply(points)
    L = torch.linalg.vector_norm(expected_results)

    avg_results = avg.apply(points)

    for i in range(batch_size):
        assert torch.all(torch.abs(expected_results[i] - avg_results[i]) < 0.01 * L), f"{expected_results[i]} != {avg_results[i]}"
