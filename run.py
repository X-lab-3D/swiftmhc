#!/usr/bin/env python
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
from torch.optim.lr_scheduler import StepLR, _LRScheduler, LambdaLR
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
from tcrspec.metrics import MetricsRecord


arg_parser = ArgumentParser(description="run a TCR-spec network model")
arg_parser.add_argument("--run-id", "-r", help="name of the run and the directory to store it")
arg_parser.add_argument("--debug", "-d", help="generate debug files", action='store_const', const=True, default=False)
arg_parser.add_argument("--log-stdout", "-l", help="log to stdout", action='store_const', const=True, default=False)
arg_parser.add_argument("--pretrained-model", "-m", help="use a given pretrained model state")
arg_parser.add_argument("--workers", "-w", help="number of workers to load batches", type=int, default=5)
arg_parser.add_argument("--batch-size", "-b", help="batch size to use during training/validation/testing", type=int, default=8)
arg_parser.add_argument("--epoch-count", "-e", help="how many epochs to run during training", type=int, default=30)
arg_parser.add_argument("--affinity-tune-count", "-j", help="how many epochs to run during affinity-tune, thus with affinity prediction", type=int, default=30)
arg_parser.add_argument("--fine-tune-count", "-u", help="how many epochs to run during fine-tuning, at the end", type=int, default=30)
arg_parser.add_argument("--animate", "-a", help="id of a data point to generate intermediary pdb for", nargs="+")
arg_parser.add_argument("--lr", help="learning rate setting", type=float, default=0.001)
arg_parser.add_argument("--classification", "-c", help="do classification instead of regression", action="store_const", const=True, default=False)
arg_parser.add_argument("--pdb-output", help="store resulting pdb files in an hdf5 file", action="store_const", const=True, default=False)
arg_parser.add_argument("--test-only", "-t", help="do not train, only run tests", const=True, default=False, action='store_const')
arg_parser.add_argument("--test-subset-path", help="path to list of entry ids that should be excluded for testing", nargs="+")
arg_parser.add_argument("data_path", help="path to a hdf5 file", nargs="+")


_log = logging.getLogger(__name__)


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

    @staticmethod
    def _save_structure_to_hdf5(structure: Structure,
                                group: h5py.Group):
        """
        Save a biopython structure object as PDB in an HDF5 file.

        Args:
            structure: data to store
            group: HDF5 group to store the data to
        """

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

    def _save_structures(self, output_directory: str, data: Dict[str, torch.Tensor], output: Dict[str, torch.Tensor]):
        """
        Used to save all models (truth) and (prediction) to two hdf5 files

        Args:
            output_directory: where to store the hdf5 files under
            data: truth data
            output: model output
        """

        truth_path = os.path.join(output_directory, 'true-structures.hdf5')
        pred_path = os.path.join(output_directory, 'predicted-structures.hdf5')

        for index, id_ in enumerate(data["ids"]):
            true_structure = recreate_structure(id_,
                                                [("P", data["loop_residue_numbers"][index], data["loop_sequence_onehot"][index], data["loop_atom14_gt_positions"][index]),
                                                 ("M", data["protein_residue_numbers"][index], data["protein_sequence_onehot"][index], data["protein_atom14_gt_positions"][index])])

            with h5py.File(truth_path, 'a') as truth_file:
                id_group = truth_file.require_group(id_)
                self._save_structure_to_hdf5(true_structure, id_group)

            pred_structure = recreate_structure(id_,
                                                [("P", data["loop_residue_numbers"][index], data["loop_sequence_onehot"][index], output["final_positions"][index]),
                                                 ("M", data["protein_residue_numbers"][index], data["protein_sequence_onehot"][index], data["protein_atom14_gt_positions"][index])])

            with h5py.File(pred_path, 'a') as pred_file:
                id_group = pred_file.require_group(id_)
                self._save_structure_to_hdf5(pred_structure, id_group)


    def _snapshot(self,
                  frame_id: str,
                  model: Predictor,
                  output_directory: str,
                  data: Dict[str, torch.Tensor]):
        """
        Use the given model to predict output for the given data and store a snapshot of the output.

        Args:
            frame_id: name of the frame to store the snapshot under
            model: the model for which the output is snapshot
            output_directory: where to store the resuling animation HDF5 file
            data: input data for the model, to take snapshots for
        """

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

    def _batch(self,
               optimizer: Optimizer,
               model: Predictor,
               data: TensorDict,
               affinity_tune: bool,
               fine_tune: bool,
    ) -> Tuple[TensorDict, Dict[str, torch.Tensor]]:
        """
        Action performed when training on a single batch.
        This involves backpropagation.

        Args:
            optimizer: needed to optimize the model
            model: the model that is trained
            data: input data for the model
            affinity_tune: whether to include affinity loss in backward propagation
            fine_tune: whether to include fine tuning losses in backward propagation

        Returns:
            the losses and the output data
        """

        # set all gradients to zero
        optimizer.zero_grad()

        # get model output
        output = model(data)

        # calculate losses
        losses = get_loss(output, data, affinity_tune, fine_tune)

        # backward propagation
        loss = losses["total"]
        loss.backward()

        # only do this if necessary, when the training isn't stable
        #clip_grad_norm_(model.parameters(), 0.5)

        # optimize
        optimizer.step()

        return (losses.detach(), output)

    def _epoch(self,
               epoch_index: int,
               optimizer: Optimizer,
               model: Predictor,
               data_loader: DataLoader,
               affinity_tune: bool,
               fine_tune: bool,
               output_directory: Optional[str] = None,
               animated_data: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Do one training epoch, with backward propagation.
        The model parameters are adjusted per batch.

        Args:
            epoch_index: the number of the epoch to save the metrics to
            optimizer: needed to optimize the model
            model: the model that will be trained
            data_loader: data to insert into the model
            affinity_tune: whether to include affinity loss in backward propagation
            fine_tune: whether to include fine tuning losses in backward propagation
            output_directory: where to store the results
            animated_data: data to store snapshot structures from, for a given fraction of the batches.
        """

        # start with an empty metrics record
        record = MetricsRecord()

        # put model in train mode
        model.train()

        for batch_index, batch_data in enumerate(data_loader):

            # Do the training step.
            batch_loss, batch_output = self._batch(optimizer, model,
                                                   batch_data,
                                                   affinity_tune, fine_tune)

            if output_directory is not None:

                # make the snapshot, if requested
                if animated_data is not None and (batch_index + 1) % self._snap_period == 0:

                    self._snapshot(f"{epoch_index}.{batch_index + 1}",
                                   model,
                                   output_directory, animated_data)

                # Save to metrics
                record.add_batch(batch_loss, batch_output, batch_data)

        # Create the metrics row for this epoch
        if output_directory is not None:
            record.save(epoch_index, data_loader.dataset.name, output_directory)

    def _validate(self,
                  epoch_index: int,
                  model: Predictor,
                  data_loader: DataLoader,
                  output_directory: Optional[str] = None,
                  save_structures: Optional[bool] = False,
    ) -> float:
        """
        Run an evaluation of the model, thus no backward propagation.
        The model parameters stay the same in this call.

        Args:
            epoch_index: the number of the epoch to save the metrics to
            model: the model that will be evaluated
            data_loader: data to insert into the model
            output_directory: where to store the results
        Returns:
            the mean total loss for all inserted data
        """

        # start with an empty metrics record
        record = MetricsRecord()

        # put model in evaluation mode
        model.eval()

        datapoint_count = 0
        sum_of_losses = 0.0

        with torch.no_grad():

            for batch_index, batch_data in enumerate(data_loader):

                # make the model generate output
                batch_output = model(batch_data)

                # calculate the losses, for monitoring only
                batch_loss = get_loss(batch_output, batch_data, True, True)

                # count the number of loss values
                datapoint_count += batch_data['loop_aatype'].shape[0]
                sum_of_losses += batch_loss['total'].item() * batch_data['loop_aatype'].shape[0]

                if save_structures and output_directory is not None:
                    self._save_structures(output_directory, batch_data, batch_output)

                # Save to metrics
                record.add_batch(batch_loss, batch_output, batch_data)

        # Create the metrics row for this epoch
        if output_directory is not None:
            record.save(epoch_index, data_loader.dataset.name, output_directory)

        return (sum_of_losses / datapoint_count)

    def test(self,
             test_loaders: [DataLoader],
             run_id: str,
             animated_complex_ids: List[str],
             model_path: str,
    ):
        """
        Call this function instead of train, when you just want to test the model.
        It saves the generated structures to hdf5 files.

        Args:
            test_loaders: test datasets to run the model on
            run_id: run directory, where the model file is stored
            animated_complex_ids: list of complex names, of which the structures should be saved
            model_path: pretrained model to use
        """

        # init model
        model = Predictor(self.loop_maxlen,
                          self.protein_maxlen,
                          self._model_type,
                          openfold_config.model)
        model = DataParallel(model)
        model.to(device=self._device)
        model.eval()

        # load the pretrained model
        model.load_state_dict(torch.load(model_path,  map_location=self._device))

        for test_loader in test_loaders:

            # run the model to output results
            self._validate(-1, model, test_loader, run_id, save_structures=True)

    @staticmethod
    def _get_selection_data_batch(datasets: List[ProteinLoopDataset], names: List[str]) -> Dict[str, torch.Tensor]:
        """
        Searches for the requested entries in the datasets and collates them together in a batch.

        Args:
            datasets: all datasets to search through
            names: the entries to search for
        Returns:
            the data batch
        Raises:
            ValueError if a single entry cannot be found
        """

        entries = []
        for name in names:
            for dataset in datasets:
                if dataset.has_entry(name):
                    entries.append(dataset.get_entry(name))
                    break
            else:
                dataset_name_s = ",".join([dataset.name for dataset in datasets])
                raise ValueError(f"entry {name} not found in datasets: {dataset_name_s}")

        return ProteinLoopDataset.collate(entries)

    @staticmethod
    def get_model_path(run_id: str) -> str:
        "pathname to use for storing the model"

        return f"{run_id}/best-predictor.pth"

    def train(self,
              train_loader: DataLoader,
              valid_loader: DataLoader,
              test_loaders: List[DataLoader],
              epoch_count: int, affinity_tune_count: int, fine_tune_count: int,
              run_id: Optional[str] = None,
              pretrained_model_path: Optional[str] = None,
              animated_complex_ids: Optional[List[str]] = None,
    ):
        """
        Call this method for training a model

        Args:
            train_loader: dataset for training
            valid_loader: dataset for validation, selecting the best model and deciding on early stopping
            test_loaders: datasets for testing on
            epoch_count: number of epochs to run optimizing just fape and chi for binding peptides
            affinity_tune_count: number of epochs to run optimizing fape and chi for binding peptides and affinity prediction
            fine_tune_count: number of epochs to run optimizing everything
            run_id: directory to store the resulting files
            pretrained_model_path: an optional pretrained model file to start from
            animated_complex_ids: names of complexes to animate during the run. their structure snapshots will be stored in an HDF5
        """

        # Set up the model
        model = Predictor(self.loop_maxlen,
                          self.protein_maxlen,
                          self._model_type,
                          openfold_config.model)
        model = DataParallel(model)
        model.to(device=self._device)

        # continue on a pretrained model, if provided
        if pretrained_model_path is not None:
            model.load_state_dict(torch.load(pretrained_model_path,
                                             map_location=self._device))

        # set the trajectory for optimization.
        # LR is reduced after a given number of epochs
        def lr_lambda(epoch_index: int):
            lr = self._lr
            if epoch_index >= epoch_count:
                lr *= 0.1

            if epoch_index >= (epoch_count + affinity_tune_count):
                lr *= 0.1

            return lr

        optimizer = Adam(model.parameters(), lr=self._lr)
        #scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        # define model paths
        model_path = f"{run_id}/best-predictor.pth"

        # Keep track of the lowest loss value seen during the run.
        lowest_loss = float("inf")

        # make the initial snapshots for animation, if we're animating something
        animated_data = None
        if animated_complex_ids is not None:

            datasets = [train_loader.dataset, valid_loader.dataset] + [test_loader.dataset for test_loader in test_loaders]
            animated_data = self._get_selection_data_batch(datasets, animated_complex_ids)

            self._snapshot("0.0",
                           model,
                           run_id, animated_data)

        # do the actual learning iteration
        total_epochs = epoch_count + affinity_tune_count + fine_tune_count
        for epoch_index in range(total_epochs):

            # flip this setting after the given number of epochs
            affinity_tune = (epoch_index >= epoch_count)
            fine_tune = (epoch_index >= (epoch_count + affinity_tune_count))

            # train during epoch
            with Timer(f"train epoch {epoch_index}") as t:
                self._epoch(epoch_index, optimizer, model, train_loader, affinity_tune, fine_tune,
                            run_id, animated_data)
                t.add_to_title(f"on {len(train_loader.dataset)} data points")

            # validate
            with Timer(f"valid epoch {epoch_index}") as t:
                valid_loss = self._validate(epoch_index, model, valid_loader, run_id)
                t.add_to_title(f"on {len(valid_loader.dataset)} data points")

            # test
            for test_loader in test_loaders:
                with Timer(f"test epoch {epoch_index}") as t:
                    self._validate(epoch_index, model, test_loader, run_id)
                    t.add_to_title(f"on {len(test_loader.dataset)} data points")

            # early stopping, if no more improvement
            if abs(valid_loss - lowest_loss) < self._early_stop_epsilon:
                break

            # If the loss improves, save the model.
            if valid_loss < lowest_loss:
                lowest_loss = valid_loss

                torch.save(model.state_dict(), model_path)
            # else:
            #    model.load_state_dict(torch.load(model_path))

            #scheduler.step(epoch=epoch_index)

    def get_data_loader(self,
                        data_path: str,
                        batch_size: int,
                        device: torch.device,
                        entry_ids: Optional[List[str]] = None) -> DataLoader:
        """
        Builds a data loader from a hdf5 dataset path.
        Args:
            data_path: HDF5 path to load
            batch_size: number of data points per batch, that the loader should output
            device: to load the batch data on
            entry_ids: an optional list of datapoint names to use, if omitted load all data.
        Returns:
            a data loader, providing access to the requested data
        """

        dataset = ProteinLoopDataset(data_path, device,
                                     loop_maxlen=self.loop_maxlen,
                                     protein_maxlen=self.protein_maxlen,
                                     entry_names=entry_ids)

        loader = DataLoader(dataset,
                            collate_fn=ProteinLoopDataset.collate,
                            batch_size=batch_size,
                            num_workers=self.workers_count)

        return loader


def read_ids_from(path: str) -> List[str]:
    "reads the context of a text file as a list of data point ids"

    ids = []
    with open(path) as file_:
        for line in file_:
            ids += line.strip().split()
    return ids


def random_subdivision(ids: List[str], fraction: float) -> Tuple[List[str], List[str]]:
    "Randomly divides a list of ids in two, given a fraction [0 - 1] to take"

    n = int(round(fraction * len(ids)))

    shuffled = torch.randperm(ids)

    return shuffled[:-n], shuffled[-n:]


def get_excluded(names_from: List[str], names_exclude: List[str]) -> List[str]:
    """
    Remove the names from one list from the other list
    Args:
        names_from: list from which names should be removed
        names_exclude: list of names that should be removed
    Returns:
        the list of remaining names
    """

    remaining_names = []
    for name in names_from:
        if name not in names_exclude:
            remaining_names.append(name)

    return remaining_names


if __name__ == "__main__":

    args = arg_parser.parse_args()

    # The commandline argument determines whether we do regression or classification.
    model_type = ModelType.REGRESSION
    if args.classification:
        model_type = ModelType.CLASSIFICATION

    # Make a directory to store the result files.
    # It must have an unique name.
    if args.run_id is not None:
        run_id = args.run_id

        suffix = 0
        while os.path.isdir(run_id):
            # if the directory already exists, add a suffix to the name
            suffix += 1
            run_id = f"{args.run_id}-{suffix}"
    else:
        # no name given, so make a random one
        run_id = str(uuid4())
    os.mkdir(run_id)

    # apply debug settings, if chosen so
    log_level = logging.INFO
    if args.debug:
        log_level = logging.DEBUG
        torch.autograd.set_detect_anomaly(True)

    # If the user wants to log to stdout, set it.
    # Otherwise log to a file.
    if args.log_stdout:
        logging.basicConfig(stream=sys.stdout,
                            level=log_level)
    else:
        logging.basicConfig(filename=f"{run_id}/tcrspec.log", filemode="a",
                            level=log_level)

    # If cuda is available, use it.
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        _log.debug(f"using {device_count} cuda devices")
        device = torch.device("cuda")
    else:
        _log.debug("using cpu device")
        device = torch.device("cpu")

    # Make sure we can use multiple workers:
    _log.debug(f"using {args.workers} workers")
    torch.multiprocessing.set_start_method('spawn')

    # Init trainer object with given arguments
    trainer = Trainer(device, args.workers, args.lr, model_type)

    if args.test_only:
        # We do a test, no training
        test_loaders = [trainer.get_data_loader(test_path, args.batch_size, device)
                        for test_path in args.data_path]
        trainer.test(test_loaders, run_id, args.animate, args.pretrained_model)

    else:  # train, validate, test
        if len(args.data_path) >= 3:
            # the sets are separate HDF5 files
            train_path, valid_path = args.data_path[:2]

            _log.debug(f"training on {train_path}")
            _log.debug(f"validating on {valid_path}")
            _log.debug(f"testing on {args.data_path[2:]}")

            train_loader = trainer.get_data_loader(train_path, args.batch_size, device)
            valid_loader = trainer.get_data_loader(valid_path, args.batch_size, device)
            test_loaders = [trainer.get_data_loader(test_path, args.batch_size, device)
                            for test_path in args.data_path[2:]]

        elif len(args.data_path) == 2:

            # assume that the train and validation sets are one HDF5 file, the other is the test set

            train_entry_names, valid_entry_names = random_subdivision(get_entry_names(args.data_path[0]), 0.1)
            train_loader = trainer.get_data_loader(args.data_path[0], args.batch_size, device, train_entry_names)
            valid_loader = trainer.get_data_loader(args.data_path[0], args.batch_size, device, valid_entry_names)
            test_loaders = [trainer.get_data_loader(args.data_path[1], args.batch_size, device)]

            _log.debug(f"training on {args.data_path[0]} subset")
            _log.debug(f"validating on {args.data_path[0]} subset")
            _log.debug(f"testing on {args.data_path[1]}")

        else:
            # only one hdf5 file for train, validation and test.

            if args.test_subset_path is not None:

                # If a test subset path was provided, take a fraction of the HDF5 file.
                test_entry_names = read_ids_from(args.test_subset_path)
                train_valid_entry_names = get_excluded(get_entry_names(args.data_path[0]), test_entry_names)
                train_entry_names, valid_entry_names = random_subdivision(train_valid_entry_names, 0.1)
            else:
                # Otherwise, subdivide randomly
                train_entry_names, valid_test_entry_names = random_subdivision(get_entry_names(args.data_path[0]), 0.2)
                valid_entry_names, test_entry_names = random_subdivision(valid_test_entry_names, 0.5)

            _log.debug(f"training, validating & testing on {args.data_path[0]} subsets")

            train_loader = trainer.get_data_loader(args.data_path[0], args.batch_size, device, train_entry_names)
            valid_loader = trainer.get_data_loader(args.data_path[0], args.batch_size, device, valid_entry_names)
            test_loaders = [trainer.get_data_loader(args.data_path[0], args.batch_size, device, test_entry_names)]

        # train with the composed datasets and user-provided settings
        trainer.train(train_loader, valid_loader, test_loaders,
                      args.epoch_count, args.affinity_tune_count, args.fine_tune_count,
                      run_id, args.pretrained_model,
                      args.animate)
