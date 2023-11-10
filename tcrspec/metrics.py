from typing import Dict, List
import os
import csv

import torch
import numpy
import pandas

from sklearn.metrics import roc_auc_score, matthews_corrcoef

from .models.data import TensorDict
from .loss import get_calpha_rmsd


def get_accuracy(truth: List[int], pred: List[int]) -> float:
    """
    A simple method to calculate accuracy from two equally long lists of class values
    """

    count = 0
    right = 0
    for i, t in enumerate(truth):
        p = pred[i]
        count += 1
        if p == t:
            right += 1

    return float(right) / count



class MetricsRecord:
    def __init__(self):
        self._data_len = 0
        self._losses_sum = {}
        self._rmsds = {}

        self._truth_data = {}
        self._output_data = {}

    def add_batch(self,
                  losses: Dict[str, torch.Tensor],
                  output: Dict[str, torch.Tensor],
                  truth: Dict[str, torch.Tensor]):
        """
        Call this once per batch, to keep track of the model's output and losses.
        """

        # count how many datapoints have passed
        batch_size = truth["loop_aatype"].shape[0]
        self._data_len += batch_size

        # add up the losses, from the given means
        for key, value in losses.items():
            if key not in self._losses_sum:
                self._losses_sum[key] = 0.0

            self._losses_sum[key] += value * batch_size

        # store the rmsd per data point
        self._rmsds.update(get_calpha_rmsd(output, truth))

        # store the affinity predictions and truth values per data point
        for key in ["affinity", "class", "classification"]:
            if key in output:
                if key not in self._output_data:
                    self._output_data[key] = []

                self._output_data[key] += output[key].cpu().tolist()

            if key in truth:
                if key not in self._truth_data:
                    self._truth_data[key] = []

                self._truth_data[key] += truth[key].cpu().tolist()

    def save(self, epoch_number: int, pass_name: str, directory_path: str):
        """
        Call this when all batches have passed, to save the resulting metrics.

        Args:
            epoch_number: to indicate at which epoch row it should be stored
            pass_name: can be train/valid/test or other
            directory_path: a directory where to store the files
        """

        self._store_individual_rmsds(pass_name, directory_path)
        self._store_metrics_table(epoch_number, pass_name, directory_path)

    def _store_individual_rmsds(self, pass_name: str, directory_path: str):

        # store to this file
        rmsds_path = os.path.join(directory_path, f"{pass_name}-rmsds.csv")

        # create a table
        ids = list(self._rmsds.keys())
        rmsd = [self._rmsds[id_] for id_ in ids]
        table_dict = {"ID": ids, "RMSD(Å)": rmsd}
        table = pandas.DataFrame(table_dict)

        # save to file
        table.to_csv(rmsds_path, sep=',', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)

    def _store_metrics_table(self, epoch_number: int, pass_name: str, directory_path: str):

        # store to this file
        metrics_path = os.path.join(directory_path, "metrics.csv")

        # load previous data from table file
        table = pandas.DataFrame(data={"epoch": [epoch_number]})
        if os.path.isfile(metrics_path):
            table = pandas.read_csv(metrics_path, sep=',')

        # make sure the table has a row for this epoch
        if epoch_number not in table["epoch"].values:
            row = pandas.DataFrame()
            for key in table:
                row[key] = [None]
            row["epoch"] = [epoch_number]

            table = pandas.concat((table, row))

        row_index = (table["epoch"] == epoch_number)

        # write losses to the table
        for loss_name, loss_value in self._losses_sum.items():
            normalized = round((loss_value / self._data_len).item(), 3)

            table.loc[row_index, f"{pass_name} {loss_name} loss"] = normalized

        # write rmsd
        mean = round(numpy.mean(list(self._rmsds.values())), 3)
        table.loc[row_index, f"{pass_name} mean binders C-alpha RMSD(Å)"] = mean

        # write affinity-related metrics
        if "classification" in self._output_data and "class" in self._truth_data and len(set(self._truth_data["class"])) > 1:
            auc = roc_auc_score(self._truth_data["class"], [row[1] for row in self._output_data["classification"]])
            table.loc[row_index, f"{pass_name} ROC AUC"] = round(auc, 3)

        if "class" in self._output_data and "class" in self._truth_data:
            acc = get_accuracy(self._truth_data["class"], self._output_data["class"])
            table.loc[row_index, f"{pass_name} accuracy"] = round(acc, 3)

            mcc = matthews_corrcoef(self._truth_data["class"], self._output_data["class"])
            table.loc[row_index, f"{pass_name} matthews correlation"] = round(mcc, 3)

        if "affinity" in self._output_data and "affinity" in self._truth_data:
            r = pearsonr(self._output_data["affinity"], self._truth_data["affinity"]).statistic
            table.loc[row_index, f"{pass_name} pearson correlation"] = round(r, 3)

        # store metrics
        table.to_csv(metrics_path, sep=',', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)
