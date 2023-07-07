import sys
from typing import Dict, Any

import torch

from openfold.utils.rigid_utils import Rigid


def get_batch_storage_size(batch_data: Dict[str, Any]):

    total_size = 0
    for value in batch_data.values():

        if isinstance(value, torch.Tensor):
            total_size += sys.getsizeof(value.storage().cpu())

        elif isinstance(value, Rigid):
            total_size += sys.getsizeof(value._trans.storage().cpu())
            total_size += sys.getsizeof(value._rots._rot_mats.storage().cpu())
            total_size += sys.getsizeof(value._rots._quats.storage().cpu())

        else:
            total_size += sys.getsizeof(value)

    return total_size
