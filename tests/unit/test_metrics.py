import os
from shutil import rmtree
from tempfile import mkdtemp

import torch
import numpy
import pandas

from tcrspec.metrics import MetricsRecord
from tcrspec.loss import get_calpha_rmsd


def test_metrics():

    pass_name = "unit"
    batch_size = 1
    nepoch = 2

    true_coord0 = torch.rand(batch_size, 9, 14, 3)
    true_coord1 = torch.rand(batch_size, 9, 14, 3)
    pred_coord0 = torch.rand(batch_size, 9, 14, 3)
    pred_coord1 = torch.rand(batch_size, 9, 14, 3)
    mask = torch.ones(batch_size, 9)

    loss0 = {"total": torch.tensor(2.0), "fape": torch.tensor(0.5)}
    loss1 = {"total": torch.tensor(1.0), "fape": torch.tensor(0.2)}

    output0 = {"ids": ["0"], "class": torch.tensor([1]), "classification": torch.tensor([[0.1, 0.5]]), "final_positions": pred_coord0}
    output1 = {"ids": ["1"], "class": torch.tensor([0]), "classification": torch.tensor([[0.0, -0.1]]), "final_positions": pred_coord1}

    truth0 = {"ids": ["0"], "class": torch.tensor([1]), "loop_atom14_gt_positions": true_coord0, "loop_cross_residues_mask": mask, "loop_aatype": torch.tensor([[0,0,0,0,0,0,0,0]])}
    truth1 = {"ids": ["1"], "class": torch.tensor([0]), "loop_atom14_gt_positions": true_coord1, "loop_cross_residues_mask": mask, "loop_aatype": torch.tensor([[1,1,1,1,1,1,1,1]])}

    rmsd0 = get_calpha_rmsd(output0, truth0)
    rmsd1 = get_calpha_rmsd(output1, truth1)
    mean_rmsd = round(numpy.mean(list(rmsd0.values()) + list(rmsd1.values())), 3)

    output_dir = mkdtemp()
    try:
        for epoch in range(nepoch):

            record = MetricsRecord()
            record.add_batch(loss0, output0, truth0)
            record.add_batch(loss1, output1, truth1)

            record.save(epoch, pass_name, output_dir)

        table = pandas.read_csv(os.path.join(output_dir, 'metrics.csv'))
    finally:
        rmtree(output_dir)

    for key in (f"{pass_name} mean binders C-alpha RMSD(Å)", f"{pass_name} accuracy", f"{pass_name} total loss"):
        assert table[key].dropna().shape[0] == nepoch, f"table has no complete '{key}'"

    assert table[f"{pass_name} mean binders C-alpha RMSD(Å)"][0] == mean_rmsd
    assert table[f"{pass_name} accuracy"][0] == 1.0
