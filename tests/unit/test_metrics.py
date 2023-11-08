import os
from shutil import rmtree
from tempfile import mkdtemp
from random import random

import torch
import numpy
import pandas

from tcrspec.metrics import MetricsRecord
from tcrspec.loss import get_calpha_rmsd


def test_metrics():

    pass_name = "unit"
    batch_size = 4
    n_batch = 2
    n_epoch = 2

    batch_datas = []
    pred_datas = []
    loss_datas = []
    rmsds = {}
    count_right = 0
    for batch_index in range(n_batch):
        true_coord = torch.rand(batch_size, 9, 14, 3) * 100.0
        pred_coord = torch.rand(batch_size, 9, 14, 3) * 100.0
        mask = torch.ones(batch_size, 9)

        loss_datas.append({"total": torch.tensor(random()), "fape": torch.tensor(random())})

        ids = [str(n) for n in range(batch_index * batch_size, (batch_index + 1) * batch_size)]

        pred_datas.append({
            "ids": ids,
            "classification": torch.rand(batch_size, 2),
            "final_positions": pred_coord,
        })
        pred_datas[-1]["class"] = torch.argmax(pred_datas[-1]["classification"], dim=1)

        batch_datas.append({
            "ids": ids,
            "class": torch.rand(batch_size) > 0.5,
            "loop_atom14_gt_positions": true_coord,
            "loop_cross_residues_mask": mask,
            "loop_aatype": (torch.rand(batch_size, 9) * 20).int(),
        })

        rmsds.update(get_calpha_rmsd(pred_datas[-1], batch_datas[-1]))

        count_right += (pred_datas[-1]["class"] == batch_datas[-1]["class"]).int().sum().item()

    mean_rmsd = round(numpy.mean(list(rmsds.values())), 3)
    mean_loss = round(sum([loss["total"] * batch_size for loss in loss_datas]).item() / (batch_size * n_batch), 3)
    acc = count_right / (batch_size * n_batch)

    output_dir = mkdtemp()
    try:
        for epoch_index in range(n_epoch):

            record = MetricsRecord()

            for batch_index in range(n_batch):
                record.add_batch(loss_datas[batch_index], pred_datas[batch_index], batch_datas[batch_index])

            record.save(epoch_index, pass_name, output_dir)

        table = pandas.read_csv(os.path.join(output_dir, 'metrics.csv'))
    finally:
        rmtree(output_dir)

    for key in (f"{pass_name} mean binders C-alpha RMSD(Å)", f"{pass_name} accuracy", f"{pass_name} total loss"):
        assert table[key].dropna().shape[0] == n_epoch, f"table has no complete '{key}'"

    assert table[f"{pass_name} total loss"][0] == mean_loss
    assert table[f"{pass_name} mean binders C-alpha RMSD(Å)"][0] == mean_rmsd
    assert table[f"{pass_name} accuracy"][0] == acc
