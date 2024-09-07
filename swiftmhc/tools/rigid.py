from __future__ import annotations

import torch

import openfold.utils.rigid_utils

from .quat import multiply_quat, rotate_vec_by_quat, conjugate_quat


class Rigid(openfold.utils.rigid_utils.Rigid):
    """
    Like in openfold, but overrides the apply methods to use quaternions.
    """

    @staticmethod
    def identity(
        shape: Tuple[int],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = True,
        fmt: str = "quat",

    ) -> Rigid:

        id_ = openfold.utils.rigid_utils.Rigid.identity(shape, dtype, device, requires_grad, fmt)

        return Rigid(id_.get_rots(), id_.get_trans())

    def compose_q_update_vec(self, q_update_vec: torch.Tensor) -> Rigid:
        """
            Composes the transformation with a quaternion update vector of
            shape [..., 6], where the final 6 columns represent the a, b, and
            c values of a quaternion of form (1, a, b, c) followed by a 3D
            translation.

            Args:
                q_vec: The quaternion update vector. [..., 6]
            Returns:
                The composed transformation. [...]
        """

        q_vec = q_update_vec[..., :3]
        t_vec = q_update_vec[..., 3:]

        # line 2 of AlphaFold2 Algorithm 23
        q_upd = q_vec.new_ones(list(q_vec.shape[:-1]) + [4])
        q_upd[..., 1:] = q_vec
        q_upd = torch.nn.functional.normalize(q_upd, dim=-1)

        # compose new transformation:
        new_q = multiply_quat(self.get_rots().get_quats(), q_upd)

        trans_update = rotate_vec_by_quat(self.get_rots().get_quats(), t_vec)

        new_translation = self.get_trans() + trans_update

        return Rigid(
            openfold.utils.rigid_utils.Rotation(quats=new_q, normalize_quats=False),
            new_translation
        )

    def apply(self, v: torch.Tensor) -> torch.Tensor:
        """
            Apply the current Rotation as a rotation matrix to a set of 3D
            coordinates.

            Args:
                pts:
                    A [*, 3] set of points
            Returns:
                [*, 3] rotated points
        """

        t = r.get_trans()
        q = r.get_rots().get_quats()

        return t + rotate_vec_by_quat(q, v)

    def invert_apply(self, v: torch.Tensor) -> torch.Tensor:
        """
            The inverse of the apply() method.

            Args:
                pts:
                    A [*, 3] set of points
            Returns:
                [*, 3] inverse-rotated points
        """

        t = r.get_trans()
        q = r.get_rots().get_quats()
        inv_q = conjugate_quat(q)

        return rotate_vec_by_quat(inv_q, v - t)
