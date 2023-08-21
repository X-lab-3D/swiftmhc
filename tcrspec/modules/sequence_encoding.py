from math import floor
from typing import Tuple

import torch


def mask_loop_left_center_right(length: int, maxlength: int) -> torch.Tensor:
    """
    Returns: a mask of <maxlength> elements, containing exactly <length> times True
    """

    middle = int(floor(maxlength / 2))

    mask = torch.zeros(maxlength, dtype=torch.bool)

    cap_length = 4
    if length <= 8:
        cap_length = int(floor((length - 2) / 2))

    mask[:cap_length] = True

    center_length = length - 2 * cap_length
    center_lefthalf = int(floor(center_length / 2))
    center_righthalf = center_length - center_lefthalf
    mask[middle - center_lefthalf: middle + center_righthalf] = True

    mask[-cap_length:] = True

    return mask
