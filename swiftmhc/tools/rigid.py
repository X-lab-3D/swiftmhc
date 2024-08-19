import torch

import openfold.utils.rigid_utils


class Rigid(openfold.utils.rigid_utils.Rigid):
    """
    Like in openfold, but overrides the apply methods to use quaternions.
    """

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
