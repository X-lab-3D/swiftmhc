import torch


def conjugate_quat(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)

def multiply_quat(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:

    return torch.cat(
        (
            q2[..., :1] * q1[..., 0:1] - q2[..., 1:2] * q1[..., 1:2] - q2[..., 2:3] * q1[..., 2:3] - q2[..., 3:4] * q1[..., 3:4],
            q2[..., :1] * q1[..., 1:2] + q2[..., 1:2] * q1[..., 0:1] - q2[..., 2:3] * q1[..., 3:4] + q2[..., 3:4] * q1[..., 2:3],
            q2[..., :1] * q1[..., 2:3] + q2[..., 1:2] * q1[..., 3:4] + q2[..., 2:3] * q1[..., 0:1] - q2[..., 3:4] * q1[..., 1:2],
            q2[..., :1] * q1[..., 3:4] - q2[..., 1:2] * q1[..., 2:3] + q2[..., 2:3] * q1[..., 1:2] + q2[..., 3:4] * q1[..., 0:1],
        ),
        dim=-1,
    )

def rotate_vec_by_quat(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    r = v.new_zeros(list(v.shape[:-1]) + [4])
    r[..., 1:] = v

    q_conj = conjugate_quat(q)
    r = multiply_quat(multiply_quat(q, r), q_conj)

    return r[..., 1:]
