import os
import sys
from time import time
from argparse import ArgumentParser
import logging
from uuid import uuid4
from typing import Tuple, Union, Optional, List, Dict, Set
import random
from math import log, sqrt
from multiprocessing import set_start_method

import ml_collections
import pandas
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.nn.modules.module import Module
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import StepLR, _LRScheduler
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.utils import clip_grad_norm_

from Bio.PDB.PDBIO import PDBIO

from openfold.np.residue_constants import (restype_atom14_ambiguous_atoms as openfold_restype_atom14_ambiguous_atoms,
                                           atom_types as openfold_atom_types,
                                           van_der_waals_radius as openfold_van_der_waals_radius,
                                           make_atom14_dists_bounds as openfold_make_atom14_dists_bounds,
                                           atom_order as openfold_atom_order)
from openfold.utils.feats import atom14_to_atom37 as openfold_atom14_to_atom37
from openfold.utils.loss import (violation_loss as openfold_compute_violation_loss,
                                 within_residue_violations as openfold_within_residue_violations,
                                 lddt_loss as openfold_compute_lddt_loss,
                                 compute_renamed_ground_truth as openfold_compute_renamed_ground_truth,
                                 backbone_loss as openfold_compute_backbone_loss,
                                 sidechain_loss as openfold_compute_sidechain_loss,
                                 supervised_chi_loss as openfold_supervised_chi_loss,
                                 find_structural_violations as openfold_find_structural_violations,
                                 between_residue_clash_loss as openfold_between_residue_clash_loss,
                                 between_residue_bond_loss as openfold_between_residue_bond_loss,
                                 softmax_cross_entropy as openfold_softmax_cross_entropy)
from openfold.data.data_transforms import (atom37_to_frames as openfold_atom37_to_frames,
                                           make_atom14_masks as openfold_make_atom14_masks)
from openfold.utils.rigid_utils import Rigid, Rotation
from openfold.config import config as openfold_config
from openfold.utils.tensor_utils import permute_final_dims

from tcrspec.time import Timer
from tcrspec.preprocess import preprocess
from tcrspec.dataset import ProteinLoopDataset
from tcrspec.modules.predictor import Predictor
from tcrspec.models.complex import ComplexClass
from tcrspec.models.amino_acid import AminoAcid
from tcrspec.tools.amino_acid import one_hot_decode_sequence
from tcrspec.loss import get_loss
from tcrspec.models.data import TensorDict
from tcrspec.tools.pdb import recreate_structure
from tcrspec.domain.amino_acid import amino_acids_by_one_hot_index


arg_parser = ArgumentParser(description="run a TCR-spec network model")
arg_parser.add_argument("--run-id", "-r", help="name of the run and the directory to store it")
arg_parser.add_argument("--debug", "-d", help="generate debug files", action='store_const', const=True, default=False)
arg_parser.add_argument("--pretrained-model", "-p", help="use a given pretrained model")
arg_parser.add_argument("--test-only", "-t", help="skip training and test on a pretrained model", action='store_const', const=True, default=False)
arg_parser.add_argument("--batch-size", "-b", help="batch size to use during training/validation/testing", type=int, default=8)
arg_parser.add_argument("--epoch-count", "-e", help="how many epochs to run during training", type=int, default=100)
arg_parser.add_argument("--fine-tune-count", "-u", help="how many epochs to run during fine-tuning", type=int, default=10)
arg_parser.add_argument("data_path", help="path to the train, validation & test hdf5", nargs="+")


_log = logging.getLogger(__name__)

def _calc_mcc(probabilities: torch.Tensor, targets: torch.Tensor) -> float:

    predictions = torch.argmax(probabilities, dim=1)

    tp = torch.count_nonzero(torch.logical_and(predictions, targets)).item()
    fp = torch.count_nonzero(torch.logical_and(predictions, torch.logical_not(targets))).item()
    tn = torch.count_nonzero(torch.logical_and(torch.logical_not(predictions), torch.logical_not(targets))).item()
    fn = torch.count_nonzero(torch.logical_and(torch.logical_not(predictions), targets)).item()

    mcc_numerator = tn * tp - fp * fn
    if mcc_numerator == 0:
        mcc = 0
    else:
        mcc_denominator = sqrt((tn + fn) * (fp + tp) * (tn + fp) * (fn + tp))
        mcc = mcc_numerator / mcc_denominator

    return mcc


def _calc_sensitivity(probabilities: torch.Tensor, targets: torch.Tensor) -> float:

    predictions = torch.argmax(probabilities, dim=1)

    tp = torch.count_nonzero(torch.logical_and(predictions, targets)).item()
    fn = torch.count_nonzero(torch.logical_and(torch.logical_not(predictions), targets)).item()

    return tp / (tp + fn)


def _calc_specificity(probabilities: torch.Tensor, targets: torch.Tensor) -> float:

    predictions = torch.argmax(probabilities, dim=1)

    tn = torch.count_nonzero(torch.logical_and(torch.logical_not(predictions), torch.logical_not(targets))).item()
    fp = torch.count_nonzero(torch.logical_and(predictions, torch.logical_not(targets))).item()

    return tn / (tn + fp)


def _calc_accuracy(probabilities: torch.Tensor, targets: torch.Tensor) -> float:

    predictions = torch.argmax(probabilities, dim=1)

    tp = torch.count_nonzero(torch.logical_and(predictions, targets)).item()
    tn = torch.count_nonzero(torch.logical_and(torch.logical_not(predictions), torch.logical_not(targets))).item()

    return (tp + tn) / predictions.shape[0]


def _calc_pearson_correlation_coefficient(x: torch.Tensor, y: torch.Tensor) -> float:

    x_mean = x.mean().item()
    y_mean = y.mean().item()

    nom = torch.sum((x - x_mean) * (y - y_mean)).item()
    den = sqrt(torch.sum(torch.square(x - x_mean)).item()) * sqrt(torch.sum(torch.square(y - y_mean)).item())

    if nom == 0.0:
        return 0.0

    if den == 0.0:
        return None

    return nom / den


class Trainer:
    def __init__(self,
                 device: torch.device):


        self._affinity_loss_function = MSELoss(reduction="mean").to(device=device)

        self._device = device

        self._early_stop_epsilon = 1e-6

    def _batch(self,
               epoch_index: int,
               batch_index: int,
               optimizer: Optimizer,
               model: Predictor,
               data: TensorDict,
               fine_tune: bool) -> Tuple[TensorDict, TensorDict]:

        optimizer.zero_grad()

        output = model(data)

        losses = get_loss(output, data, fine_tune)

        loss = losses["total"]

        loss.backward()

        # only do this if necessary, when the training isn't stable
        clip_grad_norm_(model.parameters(), 0.5)

        optimizer.step()

        return (losses.detach(),
                TensorDict({"affinity": output["affinity"].detach()}),
                TensorDict({"affinity": data["affinity"].detach()}))

    @staticmethod
    def _save_cross_matrix(file_path: str,
                           loop_amino_acids: List[AminoAcid],
                           struct_amino_acids: List[AminoAcid],
                           cross_weight_matrix: torch.Tensor):
        """
        Args:
            cross_weight_matrix: [struct_len, loop_len]
            loop_amino_acids: must be of loop length
            struct_amino_acids: must be of struct length
        """

        column_names = []
        for res_index, amino_acid in enumerate(loop_amino_acids):
            if amino_acid is not None:
                column_names.append(f"{res_index + 1} {amino_acid.three_letter_code}")
            else:
                column_names.append(f"{res_index + 1} <gap>")

        row_names = []
        for res_index, amino_acid in enumerate(struct_amino_acids):
            if amino_acid is not None:
                row_names.append(f"{res_index + 1} {amino_acid.three_letter_code}")
            else:
                row_names.append(f"{res_index + 1} <gap>")

        data_frame = pandas.DataFrame(cross_weight_matrix.numpy(force=True),
                                      columns=column_names, index=row_names)
        data_frame.to_csv(file_path)

    @staticmethod
    def _save_model_output(file_path: str,
                           target_name: str,
                           loop_sequence: str,
                           loop_positions: torch.Tensor,
                           probability: Union[float, torch.Tensor]):
        """
        Args:
            loop_positions: [loop_len, 3]
            probability: 1 or 2 numbers between {0.0 - 1.0}
        """

        with open(file_path, 'wt') as table_file:
            table_file.write(f"{target_name} sequence {loop_sequence} has predicted probability {probability}\n")

            for loop_index in range(loop_positions.shape[0]):
                table_file.write(f"loop residue {loop_index} is predicted at {loop_positions[loop_index]}\n")


    def _epoch(self,
               epoch_index: int,
               optimizer: Optimizer,
               model: Predictor,
               data_loader: DataLoader,
               fine_tune: bool) -> Tuple[TensorDict, TensorDict, TensorDict]:

        epoch_loss = TensorDict()
        epoch_data = TensorDict()
        epoch_output = TensorDict()

        model.train()

        total_data_size = 0
        for batch_index, batch_data in enumerate(data_loader):

            batch_size = batch_data["loop_sequence_embedding"].shape[0]

            # Do the training step.
            batch_loss, batch_output, batch_truth = self._batch(epoch_index,
                                                                batch_index, optimizer, model,
                                                                batch_data,
                                                                fine_tune)

            epoch_loss += batch_loss * batch_size
            total_data_size += batch_size
            epoch_data.append(batch_truth)
            epoch_output.append(batch_output)

        return (epoch_loss / total_data_size, epoch_data, epoch_output)

    def _validate(self,
                  model: Predictor,
                  data_loader: DataLoader,
                  fine_tune: bool,
                  pdb_output_directory: Optional[str] = None
    ) -> Tuple[TensorDict, TensorDict, TensorDict]:

        valid_loss = TensorDict()
        valid_data = TensorDict()
        valid_output = TensorDict()
        total_data_size = 0

        model.eval()

        if pdb_output_directory is not None:
            outputs_path = os.path.join(pdb_output_directory, "outputs.csv")

            # empty the file, so that it doesn't get appended to
            if os.path.isfile(outputs_path):
                os.remove(outputs_path)

        with torch.no_grad():

            for batch_index, batch_data in enumerate(data_loader):

                batch_size = batch_data["loop_sequence_embedding"].shape[0]

                batch_output = model(batch_data)

                batch_loss = get_loss(batch_output, batch_data, fine_tune)

                valid_loss += batch_loss * batch_size
                total_data_size += batch_size

                valid_data.append(TensorDict({"affinity": batch_data["affinity"]}))

                valid_output.append(TensorDict({"affinity": batch_output["affinity"]}))

                if pdb_output_directory is not None:
                    self._save_results_as_pdbs(pdb_output_directory, batch_data, batch_output)
                    self._save_outputs_as_csv(outputs_path,
                                              batch_data, batch_output)

        return (valid_loss / total_data_size, valid_data, valid_output)

    @staticmethod
    def _save_outputs_as_csv(output_path: str, batch_data: TensorDict, output_data: TensorDict):

        batch_size = batch_data["loop_sequence_embedding"].shape[0]

        if os.path.isfile(output_path):
            table = pandas.read_csv(output_path)
        else:
            table = pandas.DataFrame(data={"id": [], "loop": [], "output affinity": [], "true affinity": []})
            table.set_index("id")

        for batch_index in range(batch_size):

            id_ = batch_data["ids"][batch_index]

            loop_amino_acids = []
            for residue_index in range(len(batch_data["loop_sequence_embedding"][batch_index])):
                if batch_data["loop_len_mask"][batch_index][residue_index].item():
                    one_hot_code = batch_data["loop_sequence_embedding"][batch_index][residue_index]
                    amino_acid = amino_acids_by_one_hot_index[torch.nonzero(one_hot_code).item()]

                    loop_amino_acids.append(amino_acid)
            loop_sequence = "".join([amino_acid.one_letter_code for amino_acid in loop_amino_acids])

            row = pandas.DataFrame({"id": [id_], "loop": [loop_sequence],
                                    "output affinity": [output_data["affinity"][batch_index].item()],
                                    "true affinity": [batch_data["affinity"][batch_index].item()]})
            row.set_index("id")

            table = pandas.concat((table, row))

        table.to_csv(output_path, index=False)

    @staticmethod
    def _save_results_as_pdbs(output_directory: str, input_data: TensorDict, output_data: TensorDict):

        batch_size = input_data["loop_sequence_embedding"].shape[0]

        for index in range(batch_size):

            # only save the structure of binders
            if input_data["kd"][index] < 500.0:

                structure = recreate_structure(input_data["ids"][index],
                                               [("P", input_data["loop_sequence_embedding"][index], output_data["final_positions"][index]),
                                                ("M", input_data["protein_sequence_embedding"][index], input_data["protein_atom14_gt_positions"][index])])
                io = PDBIO()
                io.set_structure(structure)
                io.save(f"{output_directory}/{structure.id}.pdb")

    def test(self, test_loader: DataLoader, run_id: str):

        model_path = f"{run_id}/best-predictor.pth"
        model = Predictor(openfold_config.model, self._device)
        model.to(device=self._device)
        model.eval()
        model.load_state_dict(torch.load(model_path))

        test_loss, test_data, test_output = self._validate(model, test_loader, True, run_id)

        self._output_metrics(run_id, "test", -1, test_loss, test_data, test_output)

    def train(self,
              train_loader: DataLoader,
              valid_loader: DataLoader,
              test_loader: DataLoader,
              epoch_count: int, fine_tune_count: int,
              run_id: Optional[str] = None,
              pretrained_model_path: Optional[str] = None):

        # get train data affinities
        train_affinities = torch.cat([batch_data["affinity"] for batch_data in train_loader])

        # Set up the model
        model = Predictor(openfold_config.model)
        model.to(device=self._device)
        model.train()

        if pretrained_model_path is not None:
            model.load_state_dict(torch.load(pretrained_model_path,
                                             map_location=self._device))

        optimizer = Adam(model.parameters(), lr=0.01)
        # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

        # define model path
        model_path = f"{run_id}/best-predictor.pth"

        # Keep track of the lowest loss value.
        lowest_loss = float("inf")

        total_epochs = epoch_count + fine_tune_count
        for epoch_index in range(total_epochs):

            # flip this setting after the given number of epochs
            fine_tune = (epoch_index >= epoch_count)

            # train during epoch
            with Timer(f"train epoch {epoch_index}") as t:
                train_loss, train_data, train_output = self._epoch(epoch_index, optimizer, model, train_loader, fine_tune)
                t.add_to_title(f"on {train_data.size()} data points")

            # validate
            with Timer(f"valid epoch {epoch_index}") as t:
                valid_loss, valid_data, valid_output = self._validate(model, valid_loader, fine_tune)
                t.add_to_title(f"on {valid_data.size()} data points")

            # test
            with Timer(f"test epoch {epoch_index}") as t:

                pdb_directory = None
                if (epoch_index + 1) == total_epochs:  # final epoch?
                    pdb_directory = run_id

                test_loss, test_data, test_output = self._validate(model, test_loader, fine_tune, pdb_directory)
                t.add_to_title(f"on {test_output.size()} data points")

            # write the metrics
            self._output_metrics(run_id, "train", epoch_index, train_loss, train_data, train_output)
            self._output_metrics(run_id, "valid", epoch_index, valid_loss, valid_data, valid_output)
            self._output_metrics(run_id, "test", epoch_index, test_loss, test_data, test_output)

            # early stopping, if no more improvement
            if abs(valid_loss["total"] - lowest_loss) < self._early_stop_epsilon:
                if fine_tune:
                    break
                else:
                    epoch_index = epoch_count

            # If the loss improves, save the model.
            if valid_loss["total"] < lowest_loss:
                lowest_loss = valid_loss["total"]

                torch.save(model.state_dict(), model_path)
            # else:
            #    model.load_state_dict(torch.load(model_path))

            # scheduler.step()

    @staticmethod
    def _init_metrics_dataframe():

        metrics_dataframe = pandas.DataFrame(data={"epoch": [],
                                                   "train total loss": [],
                                                   "valid total loss": [],
                                                   "test total loss": [],
                                                   "train affinity correlation": [],
                                                   "valid affinity correlation": [],
                                                   "test affinity correlation": []})

        return metrics_dataframe

    def _output_metrics(self, run_id: str,
                        pass_name: str,
                        epoch_index: int,
                        losses: TensorDict,
                        data: TensorDict,
                        output: TensorDict):

        metrics_path = f"{run_id}/metrics.csv"
        if os.path.isfile(metrics_path):

            metrics_dataframe = pandas.read_csv(metrics_path, sep=',')
        else:
            metrics_dataframe = self._init_metrics_dataframe()
        metrics_dataframe.set_index("epoch")

        while epoch_index > metrics_dataframe.shape[0]:
            metrics_dataframe = metrics_dataframe.append({name: [] for name in metrics_dataframe})

        metrics_dataframe.at[epoch_index, "epoch"] = int(epoch_index)

        for loss_name in losses:
            metrics_dataframe.at[epoch_index, f"{pass_name} {loss_name} loss"] = round(losses[loss_name].item(), 3)

        pcc = _calc_pearson_correlation_coefficient(output["affinity"], data["affinity"])
        if pcc is not None:
            metrics_dataframe.at[epoch_index, f"{pass_name} affinity correlation"] = round(pcc, 3)

        metrics_dataframe.to_csv(metrics_path, sep=",", index=False)


def get_data_loader(data_path: str,
                    batch_size: int,
                    device: torch.device) -> DataLoader:

    protein_maxlen = 40
    loop_maxlen = 16

    dataset = ProteinLoopDataset(data_path, device, loop_maxlen=loop_maxlen, protein_maxlen=protein_maxlen)
    loader = DataLoader(dataset,
                        collate_fn=ProteinLoopDataset.collate, batch_size=batch_size,
                        num_workers=5)

    return loader


if __name__ == "__main__":

    args = arg_parser.parse_args()

    logging.basicConfig(filename="tcrspec.log", filemode="a",
                        level=logging.DEBUG if args.debug else logging.INFO)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        _log.debug("using cuda device")
    else:
        device = torch.device("cpu")
        _log.debug("using cpu device")

    set_start_method('spawn')

    if args.run_id is not None:
        run_id = args.run_id
    else:
        run_id = str(uuid4())

    if not os.path.isdir(run_id):
        os.mkdir(run_id)

    trainer = Trainer(device)
    if args.test_only:

        test_path = args.data_path[0]

        test_loader = get_data_loader(test_path, args.batch_size, device)

        trainer.test(test_loader, run_id=run_id)

    else:  # train & test
        if len(os.listdir(run_id)) > 0:
            raise ValueError(f"Already exists: {run_id}")

        train_path, valid_path, test_path = args.data_path

        train_loader = get_data_loader(train_path, args.batch_size, device)
        valid_loader = get_data_loader(valid_path, args.batch_size, device)
        test_loader = get_data_loader(test_path, args.batch_size, device)

        trainer.train(train_loader, valid_loader, test_loader,
                      args.epoch_count, args.fine_tune_count,
                      run_id, args.pretrained_model)
