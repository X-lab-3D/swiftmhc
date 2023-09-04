import logging
from typing import List
from math import sin, cos

import torch

from openfold.utils.rigid_utils import Rigid, Rotation

from tcrspec.operate import average_rigid


_log = logging.getLogger(__name__)


def _get_rotx_mat(a: float) -> List[List[float]]:
    return [[1.0, 0.0, 0.0],
            [0.0, cos(a), -sin(a)],
            [0.0, sin(a), cos(a)]]


def test_average_rigid():

    count = 10
    point = 100 * torch.rand(3)
    translations = torch.rand(count, 3)
    angles = torch.rand(count)
    rotx_mats = torch.tensor([_get_rotx_mat(a) for a in angles])

    rigids = Rigid(Rotation(rot_mats=rotx_mats), translations)

    avg = average_rigid(rigids, dim=0)

    mean_translation = torch.mean(translations, dim=0)
    mean_angle = torch.mean(angles)
    mean_rotx_mat = torch.tensor(_get_rotx_mat(mean_angle))
    expected_result = Rigid(Rotation(rot_mats=mean_rotx_mat), mean_translation).apply(point)
    L = torch.linalg.vector_norm(expected_result)

    avg_result = avg.apply(point)

    assert torch.all(torch.abs(expected_result - avg_result) < 0.01 * L), f"{expected_result} != {avg_result}"
