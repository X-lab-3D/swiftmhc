import logging
from typing import Optional, Union, List

import torch

from openfold.utils.rigid_utils import Rigid, Rotation


_log = logging.getLogger(__name__)


def average_quats(input_: torch.Tensor,
                  dim: Optional[Union[int, List[int]]] = None,
                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    based on:
    Markley et al., Averaging Quaternions, Journal of Guidance, Control, and Dynamics, 30(4):1193-1196, June 2007, Equations 12 and 13.
    """

    if dim is None:
        dim = list(range(len(input_.shape) - 1))

    quat_dim = len(input_.shape) - 1

    batched_input = input_.reshape(-1, 4)
    matrices = torch.bmm(batched_input.unsqueeze(-1), batched_input.unsqueeze(-2))
    matrices = matrices.reshape(list(input_.shape[:-1]) + [4, 4])

    if mask is not None:
        matrices = matrices * mask[..., None, None]

    sum_matrix = torch.sum(matrices, dim=dim)

    eigen_values, eigen_vectors = torch.linalg.eig(sum_matrix)

    return torch.nn.functional.normalize(eigen_vectors[..., 0].real, dim=-1)


def average_rigid(input_: Rigid,
                  dim: Optional[Union[int, List[int]]] = None,
                  mask: Optional[torch.Tensor] = None) -> Rigid:
    """
    Returns the average of a series of transformations.
    """

    input_trans = input_.get_trans()
    if mask is not None:
        sum_trans = torch.sum(input_trans * mask[..., None], dim=dim)
        count_trans = torch.sum(mask[..., None], dim=dim)
        average_trans = sum_trans / count_trans
    else:
        average_trans = torch.mean(input_trans, dim=dim)

    input_quats = input_.get_rots().get_quats()
    average_quat = average_quats(input_quats, dim=dim, mask=mask)

    return Rigid(Rotation(quats=average_quat), average_trans)
