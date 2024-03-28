from typing import Dict, List, Union
import os
import csv

import torch
import numpy
import pandas

from sklearn.metrics import roc_auc_score, matthews_corrcoef
from scipy.stats import pearsonr

from .domain.amino_acid import amino_acids_by_one_hot_index
from .models.data import TensorDict
from .loss import get_calpha_rmsd, AFFINITY_BINDING_TRESHOLD


def get_sequence(aatype: List[int], mask: List[bool]) -> str:
    """
    Converts aatype tensor to a one letter encoded sequence string.
    """

    s = ""
    for i, b in enumerate(mask):
        if b:
            s += amino_acids_by_one_hot_index[int(aatype[i])].one_letter_code

    return s


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

    batch_write_interval = 25

    def __init__(self, epoch_number: int, pass_name: str, directory_path: str):
        """
        Args:
            epoch_number: to indicate at which epoch row it should be stored
            pass_name: can be train/valid/test or other
            directory_path: a directory where to store the files
        """

        self._data_len = 0
        self._losses_sum = {}
        self._rmsds = {}
        self._peptide_sequences = {}

        self._id_order = []
        self._truth_data = {}
        self._output_data = {}

        self._epoch_number = epoch_number
        self._pass_name = pass_name
        self._directory_path = directory_path

        self._batches_passed = 0

    def add_batch(self,
                  losses: Dict[str, torch.Tensor],
                  output: Dict[str, torch.Tensor],
                  truth: Dict[str, torch.Tensor]):
        """
        Call this once per batch, to keep track of the model's output and losses.
        """

        # count how many datapoints have passed
        batch_size = truth["peptide_aatype"].shape[0]
        self._data_len += batch_size

        # memorize the order of the ids
        self._id_order += truth["ids"]

        # add up the losses, from the given means
        for key, value in losses.items():
            if key not in self._losses_sum:
                self._losses_sum[key] = 0.0

            self._losses_sum[key] += value.detach().item() * batch_size

        # store the rmsd per data point
        self._rmsds.update(get_calpha_rmsd(output, truth))

        # store the affinity predictions and truth values per data point
        for key in ["affinity", "logits", "class"]:
            if key in output:
                if key not in self._output_data:
                    self._output_data[key] = []

                self._output_data[key] += output[key].cpu().tolist()

            if key in truth:
                if key not in self._truth_data:
                    self._truth_data[key] = []

                self._truth_data[key] += truth[key].cpu().tolist()

        # store the peptide sequences
        peptide_aatype = truth["peptide_aatype"].cpu().tolist()
        peptide_mask = truth["peptide_self_residues_mask"].cpu().tolist()
        for i in range(batch_size):
            id_ = truth["ids"][i]
            peptide_sequence = get_sequence(peptide_aatype[i], peptide_mask[i])
            self._peptide_sequences[id_] = peptide_sequence

        # store the peptide sequences
        peptide_aatype = truth["peptide_aatype"].cpu().tolist()
        peptide_mask = truth["peptide_self_residues_mask"].cpu().tolist()
        for i in range(batch_size):
            id_ = truth["ids"][i]
            peptide_sequence = get_sequence(peptide_aatype[i], peptide_mask[i])
            self._peptide_sequences[id_] = peptide_sequence

        self._batches_passed += 1
        if self._batches_passed % self.batch_write_interval == 0:

            self._store_individual_rmsds(self._pass_name, self._directory_path)
            self._store_inidividual_affinities(self._pass_name, self._directory_path)

    def save(self):
        """
        Call this when all batches have passed, to save the resulting metrics.
        """

        self._store_individual_rmsds(self._pass_name, self._directory_path)
        self._store_inidividual_affinities(self._pass_name, self._directory_path)
        self._store_metrics_table(self._epoch_number, self._pass_name, self._directory_path)

    def _store_individual_rmsds(self, pass_name: str, directory_path: str):
        """
        Store the C-alpha RMSD file per binder peptide.
        """

        # store to this file
        rmsds_path = os.path.join(directory_path, f"{pass_name}-rmsds.csv")

        sequence_order = []
        rmsds = []
        ids = []
        for id_, rmsd in self._rmsds.items():
            sequence_order.append(self._peptide_sequences[id_])
            rmsds.append(rmsd)
            ids.append(id_)

        # create a table
        table_dict = {"ID": ids, "peptide": sequence_order, "RMSD(Å)": rmsds}
        table = pandas.DataFrame(table_dict)

        # save to file
        table.to_csv(rmsds_path, sep=',', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)

    def _store_inidividual_affinities(self, pass_name: str, directory_path: str):
        """
        Store the binding affinity (true and/or predicted) per peptide.
        """

        affinities_path = os.path.join(directory_path, f"{pass_name}-affinities.csv")

        sequence_order = []
        for id_ in self._id_order:
            sequence_order.append(self._peptide_sequences[id_])

        table_dict = {"ID": self._id_order, "peptide": sequence_order}

        for key in ["affinity", "class"]:
            if key in self._truth_data:
                table_dict[f"true {key}"] = self._truth_data[key]

            if key in self._output_data:
                table_dict[f"output {key}"] = self._output_data[key]

        if "affinity" in self._output_data:

            if "affinity_lt" in self._truth_data and self._truth_data["affinity_lt"]:
                table_dict["inequality"] = ["<"] * len(self._truth_data["affinity"])

            elif "affinity_gt" in self._truth_data and self._truth_data["affinity_gt"]:
                table_dict["inequality"] = [">"] * len(self._truth_data["affinity"])
            else:
                table_dict["inequality"] = ["="] * len(self._truth_data["affinity"])

        if "logits" in self._output_data:
            for i in range(2):
                table_dict[f"output logit {i}"] = [l[i] for l in self._output_data["logits"]]

        lmax = max([len(table_dict[key]) for key in table_dict])
        for key in table_dict:
            l = len(table_dict[key])
            if l != lmax:
                raise RuntimeError(f"{key} has length {l}, expecting {lmax}")

        table = pandas.DataFrame(table_dict)

        # save to file
        table.to_csv(affinities_path, sep=',', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)

    @staticmethod
    def _has_distribution(values: List[Union[float, int]]) -> bool:
        """
        if the values are all the same, returns False
        if there's at least one value different from the others, returns True
        """

        return len(set(values)) > 1

    def _store_regression(self, table: pandas.DataFrame, row_index: int, pass_name: str):
        """
        store all metrics related to regression
        """

        r = pearsonr(self._output_data["affinity"], self._truth_data["affinity"]).statistic
        table.loc[row_index, f"{pass_name} pearson correlation"] = round(r, 3)

        output_class = (numpy.array(self._output_data["affinity"]) > AFFINITY_BINDING_TRESHOLD)

        acc = get_accuracy(self._truth_data["class"], output_class)
        table.loc[row_index, f"{pass_name} accuracy"] = round(acc, 3)

        auc = roc_auc_score(self._truth_data["class"], self._output_data["affinity"])
        table.loc[row_index, f"{pass_name} ROC AUC"] = round(auc, 3)

        mcc = matthews_corrcoef(self._truth_data["class"], output_class)
        table.loc[row_index, f"{pass_name} matthews correlation"] = round(mcc, 3)

    def _store_classification(self, table: pandas.DataFrame, row_index: int, pass_name: str):
        """
        store all metrics related to classification
        """

        p = torch.nn.functional.softmax(torch.tensor(self._output_data["logits"]), dim=-1)

        auc = roc_auc_score(self._truth_data["class"], p[:, 1])
        table.loc[row_index, f"{pass_name} ROC AUC"] = round(auc, 3)

        acc = get_accuracy(self._truth_data["class"], self._output_data["class"])
        table.loc[row_index, f"{pass_name} accuracy"] = round(acc, 3)

        mcc = matthews_corrcoef(self._truth_data["class"], self._output_data["class"])
        table.loc[row_index, f"{pass_name} matthews correlation"] = round(mcc, 3)

    def _store_metrics_table(self, epoch_number: int, pass_name: str, directory_path: str):
        """
        Store the data of one epoch in the metrics table.
        """

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
            normalized = round((loss_value / self._data_len), 3)

            table.loc[row_index, f"{pass_name} {loss_name} loss"] = normalized

        # write rmsd
        mean = round(numpy.mean(list(self._rmsds.values())), 3)
        table.loc[row_index, f"{pass_name} mean binders C-alpha RMSD(Å)"] = mean

        # write affinity-related metrics
        if "class" in self._output_data and "class" in self._truth_data and self._has_distribution(self._truth_data["class"]):

            self._store_classification(table, row_index, pass_name)

        if "affinity" in self._output_data and "affinity" in self._truth_data and self._has_distribution(self._truth_data["class"]):

            self._store_regression(table, row_index, pass_name)

        # store metrics
        table.to_csv(metrics_path, sep=',', encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)
