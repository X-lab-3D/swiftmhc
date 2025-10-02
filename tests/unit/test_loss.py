# from .config import consts
import ml_collections as mlc
import torch
from openfold.utils.loss import between_residue_clash_loss as openfold_between_residue_clash_loss
from swiftmhc.loss import _between_residue_clash_loss as between_residue_clash_loss


consts = mlc.ConfigDict(
    {
        "batch_size": 2,
        "n_res": 22,
        "eps": 5e-4,
    }
)


def test_run_between_residue_clash_loss():
    bs = consts.batch_size
    n = consts.n_res
    device = torch.device("cpu")
    dtype = torch.float32

    pred_pos = torch.rand(bs, n, 14, 3, device=device, dtype=dtype).float()  # shape (2, 22, 14, 3)
    pred_atom_mask = torch.randint(
        0, 2, (bs, n, 14), device=device, dtype=dtype
    ).float()  # shape (2, 22, 14)
    atom14_atom_radius = torch.rand(
        bs, n, 14, device=device, dtype=dtype
    ).float()  # shape (2, 22, 14)

    residue_index = torch.arange(n, device=device).unsqueeze(0)  # shape (1, 22)

    between_residue_clash_loss(
        pred_pos,
        pred_atom_mask,
        atom14_atom_radius,
        residue_index,
    )


def test_run_between_residue_clash_loss_compare():
    """Compare with OpenFold implementation"""
    bs = consts.batch_size
    n = consts.n_res
    device = torch.device("cpu")
    dtype = torch.float32

    pred_pos = torch.rand(bs, n, 14, 3, device=device, dtype=dtype).float()  # shape (2, 22, 14, 3)
    pred_atom_mask = torch.randint(
        0, 2, (bs, n, 14), device=device, dtype=dtype
    ).float()  # shape (2, 22, 14)
    atom14_atom_radius = torch.rand(
        bs, n, 14, device=device, dtype=dtype
    ).float()  # shape (2, 22, 14)
    residue_index = torch.arange(n, device=device).unsqueeze(0)  # shape (1, 22)

    loss_expected = openfold_between_residue_clash_loss(
        pred_pos,
        pred_atom_mask,
        atom14_atom_radius,
        residue_index,
    )

    loss_actual = between_residue_clash_loss(
        pred_pos,
        pred_atom_mask,
        atom14_atom_radius,
        residue_index,
    )

    for k in loss_expected.keys():
        print(k)
        assert torch.allclose(loss_expected[k], loss_actual[k], atol=consts.eps)
