import os
import sys
from time import time
from argparse import ArgumentParser
import logging
from uuid import uuid4
from typing import Tuple, Union, Optional, List, Dict, Set, Any
from math import log, sqrt
import csv
import h5py
import numpy
import shutil
from io import StringIO

from sklearn.metrics import roc_auc_score, matthews_corrcoef
from scipy.stats import pearsonr
import ml_collections
import pandas
import torch
from torch.utils.data import DataLoader
from torch.nn.modules.module import Module
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import StepLR, _LRScheduler
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.utils import clip_grad_norm_
from torch.nn import DataParallel

from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Structure import Structure

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
from tcrspec.dataset import ProteinLoopDataset, get_entry_names
from tcrspec.modules.predictor import Predictor
from tcrspec.models.complex import ComplexClass
from tcrspec.models.amino_acid import AminoAcid
from tcrspec.tools.amino_acid import one_hot_decode_sequence
from tcrspec.loss import get_loss, get_calpha_rmsd
from tcrspec.models.data import TensorDict
from tcrspec.tools.pdb import recreate_structure
from tcrspec.domain.amino_acid import amino_acids_by_one_hot_index
from tcrspec.models.types import ModelType


def get_accuracy(truth: List[int], pred: List[int]) -> float:
    count = 0
    right = 0
    for i, t in enumerate(truth):
        p = pred[i]
        count += 1
        if p == t:
            right += 1

    return float(right) / count


arg_parser = ArgumentParser(description="run a TCR-spec network model")
arg_parser.add_argument("--run-id", "-r", help="name of the run and the directory to store it")
arg_parser.add_argument("--debug", "-d", help="generate debug files", action='store_const', const=True, default=False)
arg_parser.add_argument("--log-stdout", "-l", help="log to stdout", action='store_const', const=True, default=False)
arg_parser.add_argument("--pretrained-model", "-m", help="use a given pretrained model state")
arg_parser.add_argument("--workers", "-w", help="number of workers to load batches", type=int, default=5)
arg_parser.add_argument("--batch-size", "-b", help="batch size to use during training/validation/testing", type=int, default=8)
arg_parser.add_argument("--epoch-count", "-e", help="how many epochs to run during training", type=int, default=20)
arg_parser.add_argument("--affinity-tune-count", "-j", help="how many epochs to run during affinity training", type=int, default=50)
arg_parser.add_argument("--fine-tune-count", "-u", help="how many epochs to run during fine-tuning", type=int, default=10)
arg_parser.add_argument("--animate", "-a", help="id of a data point to generate intermediary pdb for", nargs="+")
arg_parser.add_argument("--lr", help="learning rate setting", type=float, default=0.001)
arg_parser.add_argument("--classification", "-c", help="do classification instead of regression", action="store_const", const=True, default=False)
arg_parser.add_argument("--pdb-output", help="store resulting pdb files in an hdf5 file", action="store_const", const=True, default=False)
arg_parser.add_argument("--test-only", "-t", help="do not train, only run tests", const=True, default=False, action='store_const')
arg_parser.add_argument("--test-subset-path", help="path to list of entry ids that should be excluded for testing", nargs="+")
arg_parser.add_argument("data_path", help="path to a hdf5 file", nargs="+")


_log = logging.getLogger(__name__)


def get_accuracy(output: List[int], truth: List[int]) -> float:

    right = 0
    for i, o in enumerate(output):
        t = truth[i]

        if o == t:
            right += 1

    return float(right) / len(output)


class Trainer:
    def __init__(self,
                 device: torch.device,
                 workers_count: int,
                 lr: float,
                 model_type: ModelType,
    ):

        self._lr = lr
        self._model_type = model_type

        self._device = device

        self._early_stop_epsilon = 1e-6

        self._snap_period = 20

        self.loop_maxlen = 16
        self.protein_maxlen = 200

        self.workers_count = workers_count

    def _batch(self,
               epoch_index: int,
               batch_index: int,
               optimizer: Optimizer,
               model: Predictor,
               data: TensorDict,
               affinity_tune: bool,
               fine_tune: bool,
    ) -> Tuple[TensorDict, Dict[str, torch.Tensor]]:

        optimizer.zero_grad()

        output = model(data)

        losses = get_loss(output, data, affinity_tune, fine_tune)

        loss = losses["total"]

        loss.backward()

        # only do this if necessary, when the training isn't stable
        #clip_grad_norm_(model.parameters(), 0.5)

        optimizer.step()

        return (losses.detach(), output)

    @staticmethod
    def _save_structure_to_hdf5(structure: Structure,
                                group: h5py.Group):
        pdbio = PDBIO()
        pdbio.set_structure(structure)
        with StringIO() as sio:

            pdbio.save(sio)

            structure_data = numpy.array([bytes(line + "\n", encoding="utf-8")
                                          for line in sio.getvalue().split('\n')
                                          if len(line.strip()) > 0],
                                         dtype=numpy.dtype("bytes"))

            group.create_dataset("structure",
                                 data=structure_data,
                                 compression="lzf")

    def _snapshot(self,
                  frame_id: str,
                  model: Predictor,
                  output_directory: str,
                  data: Dict[str, torch.Tensor]):

        # predict the animated complexes
        with torch.no_grad():
            output = model(data)

        for index, id_ in enumerate(data["ids"]):

            # one file per animated complex
            animation_path = f"{output_directory}/{id_}-animation.hdf5"

            with h5py.File(animation_path, "a") as animation_file:

                # for convenience, store the true structure in the animation file also
                if "true" not in animation_file:
                    true_group = animation_file.require_group("true")
                    structure = recreate_structure(id_,
                                                   [("P", data["loop_residue_numbers"][index], data["loop_sequence_onehot"][index], data["loop_atom14_gt_positions"][index]),
                                                    ("M", data["protein_residue_numbers"][index], data["protein_sequence_onehot"][index], data["protein_atom14_gt_positions"][index])])
                    self._save_structure_to_hdf5(structure, true_group)

                frame_group = animation_file.require_group(frame_id)

                # save predicted pdb
                structure = recreate_structure(id_,
                                               [("P", data["loop_residue_numbers"][index], data["loop_sequence_onehot"][index], output["final_positions"][index]),
                                                ("M", data["protein_residue_numbers"][index], data["protein_sequence_onehot"][index], data["protein_atom14_gt_positions"][index])])
                self._save_structure_to_hdf5(structure, frame_group)

                # save the residue numbering, for later lookup
                for key in ("protein_cross_residues_mask", "loop_cross_residues_mask",
                            "protein_residue_numbers", "loop_residue_numbers"):

                    if not key in animation_file:
                        animation_file.create_dataset(key, data=data[key][index].cpu())

    @staticmethod
    def _store_rmsds(output_directory: str, rmsds: Dict[str, float]):

        table_path = os.path.join(output_directory, 'rmsd.csv')

        # load old table first
        if os.path.isfile(table_path):
            old_table = pandas.read_csv(table_path)
            for index, row in old_table.iterrows():
                # old data does not overwrite new
                if row["ID"] not in rmsds:
                    rmsds[row["ID"]] = row["RMSD(Å)"]

        # save to file
        ids = list(rmsds.keys())
        rmsd = [rmsds[id_] for id_ in ids]
        table_dict = {"ID": ids, "RMSD(Å)": rmsd}

        table = pandas.DataFrame(table_dict)
        table.to_csv(table_path, encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)

    def _epoch(self,
               epoch_index: int,
               optimizer: Optimizer,
               model: Predictor,
               data_loader: DataLoader,
               affinity_tune: bool,
               fine_tune: bool,
               output_directory: Optional[str] = None,
               animated_data: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, Any]:

        epoch_data = {}

        model.train()

        rmsds = {}
        for batch_index, batch_data in enumerate(data_loader):

            # Do the training step.
            batch_loss, batch_output = self._batch(epoch_index,
                                                   batch_index, optimizer, model,
                                                   batch_data,
                                                   affinity_tune,
                                                   fine_tune)

            if output_directory is not None and animated_data is not None and (batch_index + 1) % self._snap_period == 0:

                self._snapshot(f"{epoch_index}.{batch_index + 1}",
                               model,
                               output_directory, animated_data)

            epoch_data = self._store_required_data(epoch_data, batch_loss, batch_output, batch_data)

            rmsds.update(get_calpha_rmsd(batch_output, batch_data))

        if output_directory is not None:
            self._store_rmsds(output_directory, rmsds)

        epoch_data["binders_c_alpha_rmsd"] = numpy.mean(list(rmsds.values()))
        return epoch_data

    def _validate(self,
                  epoch_index: int,
                  model: Predictor,
                  data_loader: DataLoader,
                  affinity_tune: bool,
                  fine_tune: bool,
                  output_directory: Optional[str] = None,
    ) -> Dict[str, Any]:

        valid_data = {}

        model.eval()

        rmsds = {}
        with torch.no_grad():

            for batch_index, batch_data in enumerate(data_loader):

                batch_size = batch_data["loop_sequence_onehot"].shape[0]

                batch_output = model(batch_data)

                batch_loss = get_loss(batch_output, batch_data, affinity_tune, fine_tune)

                valid_data = self._store_required_data(valid_data, batch_loss, batch_output, batch_data)

                rmsds.update(get_calpha_rmsd(batch_output, batch_data))

        if output_directory is not None:
            self._store_rmsds(output_directory, rmsds)

        valid_data["binders_c_alpha_rmsd"] = numpy.mean(list(rmsds.values()))

        return valid_data

    @staticmethod
    def _store_required_data(old_data: Dict[str, Any],
                             losses: Dict[str, Any],
                             output: Dict[str, Any],
                             truth: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Args:
            old_data: the old dictionary of epoch data
            losses: the dictionary of loss values for the batch
            output: the batch output
            truth: the batch truth data
        Returns: the new dictionary of epoch data

        Stores the batch data needed to generate output files.
        This will create a dictionary with epoch data, which can only
        contain a limited number of fields. (because of memory usage)
        """

        # store the old data in the new dictionary
        output_data = {}
        for key in old_data:
            output_data[key] = old_data[key]

        # determine batch size from the number of ids
        batch_size = len(truth["ids"])

        # combine the means of the losses
        for loss_name, loss_value in losses.items():
            loss_name += " loss"

            if loss_name not in output_data:
                output_data[loss_name] = loss_value.item() * batch_size
            else:
                output_data[loss_name] += loss_value.item() * batch_size

        # add data for the affinity scores
        for key in ["affinity", "classification", "class"]:
            if key in output:
                name = f"output {key}"
                if name not in output_data:
                    output_data[name] = []
                output_data[name] += output[key].cpu().tolist()

            if key in truth:
                name = f"true {key}"
                if name not in output_data:
                    output_data[name] = []
                output_data[name] += truth[key].cpu().tolist()

        # list all ids of the data points
        if "ids" not in output_data:
            output_data["ids"] = []
        output_data["ids"] += truth["ids"]

        # convert all one-hot encoded sequences to strings
        if "loop_sequence" not in output_data:
            output_data["loop_sequence"] = []
        output_data["loop_sequence"] += ["".join([aa.one_letter_code
                                                  for aa in one_hot_decode_sequence(embd)
                                                  if aa is not None])
                                         for embd in truth["loop_sequence_onehot"]]

        return output_data

    def test(self,
             test_loader: DataLoader,
             run_id: str,
             output_metrics_name: Optional[str],
             animated_complex_ids: List[str],
             model_path: Optional[str] = None,
    ):
        """
        Call this function instead of train, when you just want to test the model.

        Args:
            test_loader: test data
            run_id: run directory, where the model file is stored
            output_metrics: where to to save metrics data in a csv file
        """

        model = Predictor(self._model_type,
                          self.loop_maxlen,
                          self.protein_maxlen,
                          openfold_config.model)
        model = DataParallel(model)

        model.to(device=self._device)
        model.eval()

        # load the pretrained model
        if model_path is None:
            model_path = self.get_model_path(run_id)
        model.load_state_dict(torch.load(model_path,  map_location=self._device))

        # run the model to output results
        test_data = self._validate(-1, model, test_loader, True, True, run_id)

        # save metrics
        if output_metrics_name is not None:
            self._output_metrics(run_id, "test", -1, test_data)

        # do any requested animation snapshots
        if animated_complex_ids is not None and len(animated_complex_ids) > 0:
            animated_data = self._get_selection_data_batch([test_loader.dataset], animated_complex_ids)

            self._snapshot("test",
                           model,
                           run_id, animated_data)

    @staticmethod
    def _get_selection_data_batch(datasets: List[ProteinLoopDataset], names: List[str]) -> Dict[str, torch.Tensor]:

        entries = []
        for name in names:
            for dataset in datasets:
                if dataset.has_entry(name):
                    entries.append(dataset.get_entry(name))
                    break
            else:
                raise ValueError(f"entry not found in datasets: {name}")

        return ProteinLoopDataset.collate(entries)

    @staticmethod
    def get_model_path(run_id: str) -> str:
        return f"{run_id}/best-predictor.pth"

    def train(self,
              train_loader: DataLoader,
              valid_loader: DataLoader,
              test_loader: DataLoader,
              epoch_count: int, affinity_tune_count: int, fine_tune_count: int,
              run_id: Optional[str] = None,
              pretrained_model_path: Optional[str] = None,
              animated_complex_ids: Optional[List[str]] = None,
    ):
        # Set up the model
        model = Predictor(self.loop_maxlen,
                          self.protein_maxlen,
                          self._model_type,
                          openfold_config.model)
        model = DataParallel(model)

        model.to(device=self._device)

        if pretrained_model_path is not None:
            model.load_state_dict(torch.load(pretrained_model_path,
                                             map_location=self._device))

        optimizer = Adam(model.parameters(), lr=self._lr)
        # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

        # define model paths
        model_path = f"{run_id}/best-predictor.pth"
        if pretrained_model_path is not None:
            model_path = pretrained_model_path

        # Keep track of the lowest loss value.
        lowest_loss = float("inf")

        animated_data = None
        if animated_complex_ids is not None:
            # make snapshots for animation
            datasets = [train_loader.dataset, valid_loader.dataset, test_loader.dataset]
            animated_data = self._get_selection_data_batch(datasets, animated_complex_ids)

            self._snapshot("0.0",
                           model,
                           run_id, animated_data)

        total_epochs = epoch_count + affinity_tune_count + fine_tune_count
        for epoch_index in range(total_epochs):

            # flip this setting after the given number of epochs
            affinity_tune = (epoch_index >= epoch_count)
            fine_tune = (epoch_index >= (epoch_count + affinity_tune_count))

            # train during epoch
            with Timer(f"train epoch {epoch_index}") as t:
                train_data = self._epoch(epoch_index, optimizer, model, train_loader, affinity_tune, fine_tune,
                                         run_id, animated_data)
                t.add_to_title(f"on {len(train_loader.dataset)} data points")

            # validate
            with Timer(f"valid epoch {epoch_index}") as t:
                valid_data = self._validate(epoch_index, model, valid_loader, True, True, run_id)
                t.add_to_title(f"on {len(valid_loader.dataset)} data points")

            # test
            with Timer(f"test epoch {epoch_index}") as t:
                test_data = self._validate(epoch_index, model, test_loader, True, True, run_id)
                t.add_to_title(f"on {len(test_loader.dataset)} data points")

            # write the metrics
            self._output_metrics(run_id, "train", epoch_index, train_data)
            self._output_metrics(run_id, "valid", epoch_index, valid_data)
            self._output_metrics(run_id, "test", epoch_index, test_data)

            # early stopping, if no more improvement
            if abs(valid_data["total loss"] - lowest_loss) < self._early_stop_epsilon:
                break

            # If the loss improves, save the model.
            if valid_data["total loss"] < lowest_loss:
                lowest_loss = valid_data["total loss"]

                torch.save(model.state_dict(), model_path)
            # else:
            #    model.load_state_dict(torch.load(model_path))

            #scheduler.step()

        # write the output pdb models
        if output_pdb_path is not None:
            model.load_state_dict(torch.load(model_path, map_location=self._device))

            for loader in [train_loader, valid_loader, test_loader, structures_loader]:
                if loader is not None:
                    self._validate(-1, model, loader, True, output_pdb_path)

    @staticmethod
    def _init_metrics_dataframe():

        metrics_dataframe = pandas.DataFrame(data={"epoch": [],
                                                   "train total loss": [],
                                                   "valid total loss": [],
                                                   "test total loss": []})
        return metrics_dataframe

    def _output_metrics(self, run_id: str,
                        pass_name: str,
                        epoch_index: int,
                        data: Dict[str, Any]):

        # load any previous versions of the table
        metrics_path = f"{run_id}/metrics.csv"
        if os.path.isfile(metrics_path):

            metrics_dataframe = pandas.read_csv(metrics_path, sep=',')
        else:
            metrics_dataframe = self._init_metrics_dataframe()
        metrics_dataframe.set_index("epoch")

        # add a new row for this epoch
        while epoch_index > metrics_dataframe.shape[0]:
            metrics_dataframe = metrics_dataframe.append({name: [] for name in metrics_dataframe})

        metrics_dataframe.at[epoch_index, "epoch"] = int(epoch_index)

        # write losses to the table
        for loss_name in filter(lambda s: s.endswith(" loss"), data.keys()):

            normalized_loss = data[loss_name] / len(data["ids"])

            metrics_dataframe.at[epoch_index, f"{pass_name} {loss_name}"] = round(normalized_loss, 3)

        # write RMSD to the table
        metrics_dataframe.at[epoch_index, f"{pass_name} binders C-alpha RMSD"] = round(data["binders_c_alpha_rmsd"], 3)

        # write affinity-related metrics
        if "output classification" in data and "true class" in data and len(set(data["true class"])) > 1:
            auc = roc_auc_score(data["true class"], [row[1] for row in data["output classification"]])
            metrics_dataframe.at[epoch_index, f"{pass_name} ROC AUC"] = round(auc, 3)

        if "output class" in data and "true class" in data:
            acc = get_accuracy(data["true class"], data["output class"])
            metrics_dataframe.at[epoch_index, f"{pass_name} accuracy"] = round(acc, 3)

            mcc = matthews_corrcoef(data["true class"], data["output class"])
            metrics_dataframe.at[epoch_index, f"{pass_name} matthews correlation"] = round(mcc, 3)

        if "output affinity" in data and "true affinity" in data:
            r = pearsonr(data["output affinity"], data["true affinity"]).statistic
            metrics_dataframe.at[epoch_index, f"{pass_name} pearson correlation"] = round(r, 3)

        # save
        metrics_dataframe.to_csv(metrics_path, sep=",", index=False)

    def get_data_loader(self,
                        data_path: str,
                        batch_size: int,
                        device: torch.device,
                        entry_ids: Optional[List[str]] = None) -> DataLoader:

        dataset = ProteinLoopDataset(data_path, device,
                                     loop_maxlen=self.loop_maxlen,
                                     protein_maxlen=self.protein_maxlen,
                                     entry_names=entry_ids)

        loader = DataLoader(dataset,
                            collate_fn=ProteinLoopDataset.collate,
                            batch_size=batch_size,
                            num_workers=self.workers_count)

        return loader


    def store_entry_names(self, run_id: str, subset_name: str, entry_names: List[str]):
        with open(os.path.join(run_id, f"{subset_name}-entry-names.txt"), 'wt') as output_file:
            for entry_name in entry_names:
                output_file.write(f"{entry_name}\n")


def read_ids_from(path: str) -> List[str]:
    ids = []
    with open(path) as file_:
        for line in file_:
            ids += line.strip().split()
    return ids


def random_subdivision(ids: List[str], fraction: float) -> Tuple[List[str], List[str]]:

    n = int(round(fraction * len(ids)))

    shuffled = torch.randperm(ids)

    return shuffled[:-n], shuffled[-n:]


def get_excluded(names_from: List[str], names_exclude: List[str]) -> List[str]:
    remaining_names = []
    for name in names_from:
        if name not in names_exclude:
            remaining_names.append(name)

    return remaining_names


if __name__ == "__main__":

    args = arg_parser.parse_args()

    model_type = ModelType.REGRESSION
    if args.classification:
        model_type = ModelType.CLASSIFICATION

    if args.run_id is not None:
        run_id = args.run_id

        suffix = 0
        while os.path.isdir(run_id):
            suffix += 1
            run_id = f"{args.run_id}-{suffix}"
    else:
        run_id = str(uuid4())

    os.mkdir(run_id)

    log_level = logging.INFO
    if args.debug:
        log_level = logging.DEBUG
        torch.autograd.set_detect_anomaly(True)

    if args.log_stdout:
        logging.basicConfig(stream=sys.stdout,
                            level=log_level)
    else:
        logging.basicConfig(filename=f"{run_id}/tcrspec.log", filemode="a",
                            level=log_level)

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        _log.debug(f"using {device_count} cuda devices")
        device = torch.device("cuda")
    else:
        _log.debug("using cpu device")
        device = torch.device("cpu")

    if args.debug:
        torch.autograd.detect_anomaly(True)

    _log.debug(f"using {args.workers} workers")
    torch.multiprocessing.set_start_method('spawn')

    trainer = Trainer(device, args.workers, args.lr, model_type)
    model_path = trainer.get_model_path(run_id)

    if args.test_only:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)

        for test_path in args.data_path:
            data_name = os.path.splitext(os.path.basename(test_path))[0]
            test_loader = trainer.get_data_loader(test_path, args.batch_size, device)
            trainer.test(test_loader, run_id, data_name, args.animate, pdb_output_path)

    else:  # train, validate, test

        if len(args.data_path) >= 3:
            train_path, valid_path, test_path = args.data_path[:3]

            train_loader = trainer.get_data_loader(train_path, args.batch_size, device)
            valid_loader = trainer.get_data_loader(valid_path, args.batch_size, device)
            test_loader = trainer.get_data_loader(test_path, args.batch_size, device)

        elif len(args.data_path) == 2:

            train_entry_names, valid_entry_names = random_subdivision(get_entry_names(args.data_path[0]), 0.1)
            train_loader = trainer.get_data_loader(args.data_path[0], args.batch_size, device, train_entry_names)
            valid_loader = trainer.get_data_loader(args.data_path[0], args.batch_size, device, valid_entry_names)
            test_loader = trainer.get_data_loader(args.data_path[1], args.batch_size, device)

            trainer.store_entry_names(run_id, 'train', train_entry_names)
            trainer.store_entry_names(run_id, 'valid', valid_entry_names)

        else:  # only one hdf5 file

            if args.test_subset_path is not None:
                test_entry_names = read_ids_from(args.test_subset_path)
                train_valid_entry_names = get_excluded(get_entry_names(args.data_path[0]), test_entry_names)
                train_entry_names, valid_entry_names = random_subdivision(train_valid_entry_names, 0.1)
            else:
                train_entry_names, valid_test_entry_names = random_subdivision(get_entry_names(args.data_path[0]), 0.2)
                valid_entry_names, test_entry_names = random_subdivision(valid_test_entry_names, 0.5)

            train_loader = trainer.get_data_loader(args.data_path[0], args.batch_size, device, train_entry_names)
            valid_loader = trainer.get_data_loader(args.data_path[0], args.batch_size, device, valid_entry_names)
            test_loader = trainer.get_data_loader(args.data_path[0], args.batch_size, device, test_entry_names)

            trainer.store_entry_names(run_id, 'train', train_entry_names)
            trainer.store_entry_names(run_id, 'valid', valid_entry_names)
            trainer.store_entry_names(run_id, 'test', test_entry_names)

        trainer.train(train_loader, valid_loader, test_loader,
                      args.epoch_count, args.affinity_tune_count, args.fine_tune_count,
                      run_id, args.pretrained_model,
                      args.animate)
