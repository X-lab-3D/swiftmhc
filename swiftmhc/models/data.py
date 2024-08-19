from typing import Union, Dict, Optional, Set, Tuple, List
import logging

import torch

from openfold.utils.rigid_utils import Rotation

from .tools.rigid import Rigid


_log = logging.getLogger(__name__)


class TensorDict:
    """
    This is like a Dictionary of tensors, but it can undergo math operations.
    """

    def __init__(self, data: Optional[Dict[str, Union[torch.Tensor, Rigid, List[str]]]] = None):

        if data is None:
            self._data = {}
        else:
            self._data = data

    def __setitem__(self, key: str, value: Union[torch.Tensor, Rigid, List[str]]):
        self._data[key] = value

    def __getitem__(self, key: str):
        return self._data[key]

    def size(self) -> int:
        size = None
        for value in self._data.values():
            if isinstance(value, Rigid):
                item_size = value._trans.shape[0]

            elif isinstance(value, torch.Tensor) and len(value.shape) > 0:
                item_size = value.shape[0]

            elif isinstance(value, list):
                item_size = len(value)

            else:
                item_size = 1

            if size is not None and item_size != size:
                raise TypeError(f"shapes are not consistent: {size} != {item_size}")

            size = item_size

        return size

    def __iter__(self):
        return iter(self._data)

    def keys(self) -> Set[str]:
        return self._data.keys()

    def items(self) -> Set[Tuple[str, Union[torch.Tensor, Rigid, List[str]]]]:
        return self._data.items()

    def take(self, *keys):
        return TensorDict({key: self._data[key] for key in keys})

    def append(self, other):

        for key, other_value in other.items():

            if key in self._data:
                self_value = self._data[key]

                if isinstance(other_value, Rigid):
                    self._data[key] = Rigid(Rotation(torch.cat((self_value._rots._rot_mats,
                                                                other_value._rots._rot_mats), dim=0)),
                                            torch.cat((self_value._trans, other_value._trans), dim=0))

                elif isinstance(other_value, torch.Tensor):
                    self._data[key] = torch.cat((self_value, other_value), dim=0)
                else:
                    self._data[key] = self_value + other_value
            else:
                self._data[key] = other_value

    def combine(self, other):

        d = {}
        for key, value in self._data.items():
            d[key] = value

        for key, value in other._data.items():
            d[key] = value

        return TensorDict(d)

    def __add__(self, other):

        d = {}
        keys = self.keys() | other.keys()

        for key in keys:
            if key in self._data:
                d[key] = self._data[key] + other._data[key]
            else:
                d[key] = other._data[key]

        return TensorDict(d)

    def __mul__(self, scalar):

        d = {}
        for key in self._data:
            d[key] = scalar * self._data[key]

        return TensorDict(d)

    def __truediv__(self, scalar: Union[int, float]):
        d = {key: self._data[key] / scalar
             for key in self._data}

        return TensorDict(d)

    def detach(self):

        d = {}
        for key, value in self._data.items():
            if isinstance(value, Rigid):
                d[key] = Rigid(Rotation(value._rots._rot_mats.detach()),
                               value._trans.detach())

            elif isinstance(value, torch.Tensor):
                d[key] = value.detach()

            else:
                d[key] = value

        return TensorDict(d)

    def to(self, *args, **kwargs):

        d = {}
        for key, value in self._data.items():
            if isinstance(value, Rigid) or isinstance(value, torch.Tensor):
                d[key] = value.to(*args, **kwargs)
            else:
                d[key] = value

        return TensorDict(d)

    def __repr__(self):

        result = "{\n"
        for key, value in self._data.items():
            if type(value) == Rigid:
                value = value.to_tensor_4x4()
            result += f"\t{key} : {value.shape} ({value.dtype})\n"

        result += "}"

        return result
