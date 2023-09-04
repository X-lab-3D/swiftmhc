import logging
from typing import Optional, Union, List

import torch

from openfold.utils.rigid_utils import Rigid, Rotation


_log = logging.getLogger(__name__)


def average_quats(input_: torch.Tensor, dim: Optional[Union[int, List[int]]] = None) -> torch.Tensor:
    """
    based on:
    Markley et al., Averaging Quaternions, Journal of Guidance, Control, and Dynamics, 30(4):1193-1196, June 2007, Equations 12 and 13.
    """

    if dim is None:
        dim = list(range(len(input_.shape) - 1))

    preserved_dim = len(input_.shape) - 1

    sum_matrix = torch.sum(torch.mul(input_.unsqueeze(preserved_dim), input_.unsqueeze(preserved_dim + 1)), dim=dim)

    eigen_values, eigen_vectors = torch.linalg.eig(sum_matrix)

    max_index = torch.argmax(eigen_values.real)

    return torch.nn.functional.normalize(eigen_vectors[:, max_index].real, dim=0)


def average_rigid(input_: Rigid, dim: Optional[Union[int, List[int]]] = None) -> Rigid:
    """
    Returns the average of a series of transformations.
    """

    input_trans = input_.get_trans()
    average_trans = torch.mean(input_trans, dim=dim)

    input_quats = input_.get_rots().get_quats()
    average_quat = average_quats(input_quats, dim=dim)

    return Rigid(Rotation(quats=average_quat), average_trans)
