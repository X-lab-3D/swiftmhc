#!/usr/bin/env python
import logging
import os
import random
import sys
from argparse import ArgumentParser
from copy import copy
from io import StringIO
from multiprocessing.pool import Pool
from timeit import default_timer as timer
from uuid import uuid4
import h5py
import numpy
import openmm.app
import torch
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Structure import Structure
from filelock import FileLock
from openfold.np.residue_constants import restype_atom14_mask
from torch.nn import DataParallel
from torch.nn.utils import clip_grad_norm_
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer
from torch.profiler import ProfilerActivity
from torch.profiler import profile
from torch.profiler import record_function
from torch.profiler import schedule
from torch.utils.data import DataLoader
from swiftmhc.config import config as default_config
from swiftmhc.dataset import ProteinLoopDataset
from swiftmhc.dataset import get_entry_names
from swiftmhc.loss import get_loss
from swiftmhc.metrics import MetricsRecord
from swiftmhc.models.data import TensorDict
from swiftmhc.models.model_types import ModelType
from swiftmhc.modules.predictor import Predictor
from swiftmhc.time import Timer
from swiftmhc.tools.md import build_modeller
from swiftmhc.tools.md import minimize
from swiftmhc.tools.pdb import recreate_structure
from swiftmhc.tools.train import EarlyStopper
from swiftmhc.tools.train import TrainingPhase


arg_parser = ArgumentParser(description="train/test a SwiftMHC network model")
arg_parser.add_argument("--run-id", "-r", help="name of the run and the directory to store it")
arg_parser.add_argument(
    "--debug", "-d", help="generate debug files", action="store_const", const=True, default=False
)
arg_parser.add_argument(
    "--log-stdout", "-l", help="log to stdout", action="store_const", const=True, default=False
)
arg_parser.add_argument("--pretrained-model", "-p", help="use a given pretrained model state")
arg_parser.add_argument(
    "--workers", "-w", help="number of worker processes to load batches", type=int, default=5
)
arg_parser.add_argument(
    "--builders",
    "-B",
    help="number of simultaneous structure builder processes, it has no effect setting this higher than the number of models produced per batch",
    type=int,
    default=1,
)
arg_parser.add_argument(
    "--batch-size",
    "-b",
    help="batch size to use during training/validation/testing",
    type=int,
    default=20,
)
arg_parser.add_argument(
    "--phase-1-epoch-count",
    "-1",
    help="how many epochs to run during phase 1 training (structure + BA loss)",
    type=int,
    default=1000,
)
arg_parser.add_argument(
    "--phase-2-epoch-count",
    "-2",
    help="how many epochs to run during phase 2 training (structure + BA loss + fine tuning loss)",
    type=int,
    default=100,
)
arg_parser.add_argument(
    "--animate", "-m", help="id of a data point to generate intermediary pdb for", nargs="+"
)
arg_parser.add_argument(
    "--with-energy-minimization",
    help="include structure OpenMM energy minimization",
    action="store_const",
    const=True,
    default=False,
)
arg_parser.add_argument(
    "--disable-ba-loss",
    help="disable BA loss term",
    action="store_const",
    const=True,
    default=False,
)
arg_parser.add_argument(
    "--disable-struct-loss",
    help="disable structural loss terms",
    action="store_const",
    const=True,
    default=False,
)
arg_parser.add_argument(
    "--enable-compile",
    help="enable PyTorch compile during training",
    action="store_const",
    const=True,
    default=False,
)
arg_parser.add_argument(
    "--enable-profiling",
    help="enable PyTorch profiler during training",
    action="store_const",
    const=True,
    default=False,
)
arg_parser.add_argument(
    "--profile-wait",
    help="profiler schedule wait steps",
    type=int,
    default=3,
)
arg_parser.add_argument(
    "--profile-warmup",
    help="profiler schedule warmup steps",
    type=int,
    default=2,
)
arg_parser.add_argument(
    "--profile-active",
    help="profiler schedule active steps",
    type=int,
    default=1,
)
arg_parser.add_argument(
    "--profile-repeat",
    help="profiler schedule repeat count",
    type=int,
    default=1,
)
arg_parser.add_argument(
    "--no-profile-shapes",
    help="disable recording shapes in profiler",
    action="store_const",
    const=True,
    default=False,
)
arg_parser.add_argument(
    "--no-profile-memory",
    help="disable memory profiling",
    action="store_const",
    const=True,
    default=False,
)
arg_parser.add_argument(
    "--no-profile-stack",
    help="disable stack trace recording in profiler",
    action="store_const",
    const=True,
    default=False,
)
arg_parser.add_argument(
    "--classification",
    "-c",
    help="do classification instead of regression",
    action="store_const",
    const=True,
    default=False,
)
arg_parser.add_argument(
    "--test-only",
    "-t",
    help="do not train, only run tests",
    const=True,
    default=False,
    action="store_const",
)
arg_parser.add_argument(
    "--test-subset-path", help="path to list of entry ids that should be excluded for testing"
)
arg_parser.add_argument("--patience", help="early stopping patience", type=int, default=50)
arg_parser.add_argument("data_path", help="path to a hdf5 file", nargs="+")


_log = logging.getLogger(__name__)


def save_structure_to_hdf5(structure: Structure, group: h5py.Group):
    """Save a biopython structure object as PDB in an HDF5 file.

    Args:
        structure: data to store
        group: HDF5 group to store the data to
    """
    with StringIO() as sio:
        pdbio = PDBIO()
        pdbio.set_structure(structure)
        pdbio.save(sio)

        structure_data = numpy.array(
            [
                bytes(line + "\n", encoding="utf-8")
                for line in sio.getvalue().split("\n")
                if len(line.strip()) > 0
            ],
            dtype=numpy.dtype("bytes"),
        )

    group.create_dataset("structure", data=structure_data, compression="lzf")


def recreate_structure_to_hdf5(
    hdf5_path: str,
    name: str,
    data: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
):
    """Builds a structure from the atom coordinates and saves it to a hdf5 file

    Args:
        hdf5_path: output file
        name: hdf5 group name to store under
        data: structure data, to create structure from
    """
    _log.debug(f"recreating {name}")

    try:
        structure = recreate_structure(name, data)

        with FileLock(f"{hdf5_path}.lock"):
            with h5py.File(hdf5_path, "a") as hdf5_file:
                name_group = hdf5_file.require_group(name)
                save_structure_to_hdf5(structure, name_group)
    except:
        _log.exception("on recreating structure")


def minimize_structure_to_hdf5(
    hdf5_path: str,
    name: str,
    data: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
):
    """Builds a structure from the atom coordinates, minimizes it using OpenMM and saves it to a hdf5 file

    Args:
        hdf5_path: output file
        name: hdf5 group name to store under
        data: structure data, to create structure from
    """
    _log.debug(f"minimizing {name}")

    try:
        modeller = build_modeller(data)
        modeller = minimize(modeller)

        io = StringIO()

        openmm.app.PDBFile.writeFile(modeller.topology, modeller.getPositions(), io, keepIds=True)

        structure_data = numpy.array(
            [
                bytes(line + "\n", encoding="utf-8")
                for line in io.getvalue().split("\n")
                if len(line.strip()) > 0
            ],
            dtype=numpy.dtype("bytes"),
        )

        with FileLock(f"{hdf5_path}.lock"):
            with h5py.File(hdf5_path, "a") as hdf5_file:
                group = hdf5_file.require_group(name)
                group.create_dataset("structure", data=structure_data, compression="lzf")
    except:
        _log.exception(f"on recreating structure {name}")


def output_structures_to_directory(
    process_count: int,
    output_directory: str,
    filename_prefix: str,
    data: dict[str, torch.Tensor],
    output: dict[str, torch.Tensor],
    minimize_energy: bool,
):
    """Used to save all models (truth) and (prediction) to two hdf5 files

    Args:
        process_count: how many simultaneous processes to run, writing PDB to disk
        output_directory: where to store the hdf5 files under
        filename_prefix: prefix to put in the hdf5 filename
        data: truth data, from dataset
        output: model output
        minimize_energy: minimize structure or not
    """
    # define paths
    truth_path = os.path.join(output_directory, f"{filename_prefix}-true.hdf5")
    pred_path = os.path.join(output_directory, f"{filename_prefix}-predicted.hdf5")

    # don't need more processes than structures
    if process_count > len(data["ids"]):
        process_count = len(data["ids"])

    _log.debug(f"output structures to {output_directory}")

    # use a pool here, because the conversion from tensor to pdb format can be time consuming.
    with Pool(process_count) as pool:
        # get the data from gpu to cpu, if not already
        classes = data["class"].cpu()

        peptide_residue_numbers = data["peptide_residue_numbers"].cpu()
        peptide_aatype = data["peptide_aatype"].cpu()

        peptide_atom14_positions = output["final_positions"].cpu()
        peptide_mask = data["peptide_self_residues_mask"].cpu()
        peptide_atom14_mask = (
            peptide_mask.new_tensor(restype_atom14_mask[peptide_aatype]) * peptide_mask[..., None]
        )

        protein_residue_numbers = data["protein_residue_numbers"].cpu()
        protein_aatype = data["protein_aatype"].cpu()
        protein_atom14_gt_positions = data["protein_atom14_gt_positions"].cpu()
        protein_atom14_exists = data["protein_atom14_gt_exists"].cpu()

        for index, id_ in enumerate(data["ids"]):
            # do binders only
            if classes[index].item() == 0:
                continue

            # submit job for output structure
            if minimize_energy:
                pool.apply_async(
                    minimize_structure_to_hdf5,
                    (
                        pred_path,
                        id_,
                        [
                            (
                                "P",
                                peptide_residue_numbers[index],
                                peptide_aatype[index],
                                peptide_atom14_positions[index],
                                peptide_atom14_mask[index],
                            ),
                            (
                                "M",
                                protein_residue_numbers[index],
                                protein_aatype[index],
                                protein_atom14_gt_positions[index],
                                protein_atom14_exists[index],
                            ),
                        ],
                    ),
                    error_callback=lambda e: _log.exception(f"on minimizing {id_}"),
                )
            else:
                pool.apply_async(
                    recreate_structure_to_hdf5,
                    (
                        pred_path,
                        id_,
                        [
                            (
                                "P",
                                peptide_residue_numbers[index],
                                peptide_aatype[index],
                                peptide_atom14_positions[index],
                                peptide_atom14_mask[index],
                            ),
                            (
                                "M",
                                protein_residue_numbers[index],
                                protein_aatype[index],
                                protein_atom14_gt_positions[index],
                                protein_atom14_exists[index],
                            ),
                        ],
                    ),
                    error_callback=lambda e: _log.exception(f"on recreating {id_}"),
                )

        pool.close()
        pool.join()


def remove_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Paralellisation adds a 'module.' prefix in the state's keys, remove those prefixes here."""
    d = {}
    for key, value in state.items():
        if key.startswith("module."):
            new_key = key[7:]

            if new_key in d:
                raise ValueError(f"duplicate key: {new_key}")

            d[new_key] = value
        else:
            d[key] = value

    return d


class Trainer:
    def __init__(
        self,
        device: torch.device,
        float_dtype: torch.dtype,
        workers_count: int,
        model_type: ModelType,
    ):
        """Initialize the Trainer.

        Args:
            device: will be used to load the model parameters on
            float_dtype: will be used to convert the model parameters to
            workers_count: number of workers to simultaneously read from the data
            model_type: regression or classification
        """
        self.config = default_config
        self.config.model_type = model_type

        self._device = device
        self._float_dtype = float_dtype

        # for snapshots: every 20 batches
        self._snap_period = 20

        self.workers_count = workers_count

    def get_device_count(self) -> int:
        """Counts the number of devices set"""
        if self._device.type == "cuda":
            return torch.cuda.device_count()
        else:
            return torch.cpu.device_count()

    def _snapshot(
        self, frame_id: str, model: Predictor, output_directory: str, data: dict[str, torch.Tensor]
    ):
        """Use the given model to predict output for the given data and store a snapshot of the output.

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
                    structure = recreate_structure(
                        id_,
                        [
                            (
                                "P",
                                data["peptide_residue_numbers"][index],
                                data["peptide_aatype"][index],
                                data["peptide_atom14_gt_positions"][index],
                                data["peptide_atom14_gt_exists"][index],
                            ),
                            (
                                "M",
                                data["protein_residue_numbers"][index],
                                data["protein_aatype"][index],
                                data["protein_atom14_gt_positions"][index],
                                data["protein_atom14_gt_exists"][index],
                            ),
                        ],
                    )
                    save_structure_to_hdf5(structure, true_group)

                frame_group = animation_file.require_group(frame_id)

                # save predicted pdb
                structure = recreate_structure(
                    id_,
                    [
                        (
                            "P",
                            data["peptide_residue_numbers"][index],
                            data["peptide_aatype"][index],
                            output["final_positions"][index],
                            data["peptide_atom14_gt_exists"][index],
                        ),
                        (
                            "M",
                            data["protein_residue_numbers"][index],
                            data["protein_aatype"][index],
                            data["protein_atom14_gt_positions"][index],
                            data["protein_atom14_gt_exists"][index],
                        ),
                    ],
                )
                save_structure_to_hdf5(structure, frame_group)

    def _batch(
        self,
        optimizer: Optimizer,
        model: Predictor,
        data: TensorDict,
        affinity_tune: bool,
        fape_tune: bool,
        torsion_tune: bool,
        fine_tune: bool,
    ) -> tuple[TensorDict, dict[str, torch.Tensor]]:
        """Action performed when training on a single batch.

        This involves backpropagation.

        Args:
            optimizer: needed to optimize the model
            model: the model that is trained
            data: input data for the model
            affinity_tune: whether to include affinity loss in the backward propagation
            fape_tune: whether to include fape loss in backward propagation
            torsion_tune: whether to include torsion loss in backward propagation
            fine_tune: whether to include fine tuning losses in backward propagation
        Returns:
            the losses [0] and the output data [1]
        """
        # set all gradients to zero
        optimizer.zero_grad()

        # get model output
        with record_function("FORWARD"):
            output = model(data)

        # calculate losses
        with record_function("LOSS"):
            losses = get_loss(
                self.config.model_type,
                output,
                data,
                affinity_tune,
                fape_tune,
                torsion_tune,
                fine_tune,
            )

        # backward propagation
        loss = losses["total"]

        with record_function("BACKWARD"):
            loss.backward()

        # for preventing loss spikes
        clip_grad_norm_(model.parameters(), max_norm=0.5)

        # optimize
        with record_function("OPTIMIZE"):
            optimizer.step()

        return (losses.detach(), output)

    def _epoch_train(
        self,
        epoch_index: int,
        optimizer: Optimizer,
        model: Predictor,
        data_loader: DataLoader,
        affinity_tune: bool,
        fape_tune: bool,
        torsion_tune: bool,
        fine_tune: bool,
        output_directory: str,
        animated_data: dict[str, torch.Tensor] | None,
        record: MetricsRecord,
        profiler=None,
    ):
        """Process all batches in a data loader during training.

        Args:
            epoch_index: Current epoch number
            optimizer: The optimizer for training
            model: The model being trained
            data_loader: Data loader containing the batches
            affinity_tune: Whether to include affinity loss
            fape_tune: Whether to include fape loss
            torsion_tune: Whether to include torsion loss
            fine_tune: Whether to include fine tuning losses
            output_directory: Directory for output files
            animated_data: Data for animation snapshots
            record: Metrics record to update
            profiler: Optional profiler instance to step
        """
        torch.cuda.reset_peak_memory_stats()
        dataload_time = []
        batch_time = []

        dataload_time_start = timer()
        for batch_index, batch_data in enumerate(data_loader):
            dataload_time.append(timer() - dataload_time_start)

            batch_time_start = timer()
            # Transfer batch to device
            batch_data = {
                k: v.to(self._device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch_data.items()
            }

            # Do the training step.
            batch_loss, batch_output = self._batch(
                optimizer,
                model,
                batch_data,
                affinity_tune,
                fape_tune,
                torsion_tune,
                fine_tune,
            )

            # measure time taken
            batch_time.append(timer() - batch_time_start)

            # make the snapshot, if requested
            if animated_data is not None and (batch_index + 1) % self._snap_period == 0:
                self._snapshot(
                    f"{epoch_index}.{batch_index + 1}",
                    model,
                    output_directory,
                    animated_data,
                )

            # Save to metrics
            record.add_batch(batch_loss, batch_output, batch_data)

            # Step the profiler if provided
            if profiler is not None:
                profiler.step()

            dataload_time_start = timer()

        for i in dataload_time:
            _log.info(f"data load time (s): {i:.6f}")
        _log.info(
            f"data load time stat (s): mean {numpy.mean(dataload_time):.6f}, max {numpy.max(dataload_time):.6f}, min {numpy.min(dataload_time):.6f}"
        )

        for i in batch_time:
            _log.info(f"batch time (s): {i:.6f}")
        _log.info(
            f"batch time stat (s): mean {numpy.mean(batch_time):.6f}, max {numpy.max(batch_time):.6f}, min {numpy.min(batch_time):.6f}"
        )

        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        _log.info(f"Epoch peak allocated memory: {peak_mem:.2f} GB")

        peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
        _log.info(f"Epoch peak reserved memory: {peak_reserved:.2f} GB")

    def _epoch(
        self,
        epoch_index: int,
        optimizer: Optimizer,
        model: Predictor,
        data_loader: DataLoader,
        affinity_tune: bool,
        fape_tune: bool,
        torsion_tune: bool,
        fine_tune: bool,
        output_directory: str,
        animated_data: dict[str, torch.Tensor] | None = None,
        enable_profiling: bool = False,
        profile_wait: int = 3,
        profile_warmup: int = 2,
        profile_active: int = 1,
        profile_repeat: int = 1,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = True,
    ):
        """Do one training epoch, with backward propagation.

        The model parameters are adjusted per batch.

        Args:
            epoch_index: the number of the epoch to save the metrics to
            optimizer: needed to optimize the model
            model: the model that will be trained
            data_loader: data to insert into the model
            affinity_tune: whether to include affinity loss in the backward propagation
            fape_tune: whether to include fape loss in backward propagation
            torsion_tune: whether to include torsion loss in backward propagation
            fine_tune: whether to include fine tuning losses in backward propagation
            output_directory: where to store the results
            animated_data: data to store snapshot structures from, for a given fraction of the batches.
            enable_profiling: whether to enable PyTorch profiler during training
            profile_wait: profiler schedule wait steps
            profile_warmup: profiler schedule warmup steps
            profile_active: profiler schedule active steps
            profile_repeat: profiler schedule repeat count
            record_shapes: whether to record tensor shapes in profiler
            profile_memory: whether to enable memory profiling
            with_stack: whether to record stack traces in profiler
        """
        # start with an empty metrics record
        record = MetricsRecord(epoch_index, data_loader.dataset.name, output_directory)

        # put model in train mode
        model.train()

        if enable_profiling:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(
                    wait=profile_wait,
                    warmup=profile_warmup,
                    active=profile_active,
                    repeat=profile_repeat,
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    output_directory, use_gzip=True
                ),
                record_shapes=record_shapes,
                profile_memory=profile_memory,
                with_stack=with_stack,
            ) as prof:
                self._epoch_train(
                    epoch_index,
                    optimizer,
                    model,
                    data_loader,
                    affinity_tune,
                    fape_tune,
                    torsion_tune,
                    fine_tune,
                    output_directory,
                    animated_data,
                    record,
                    prof,
                )
        else:
            self._epoch_train(
                epoch_index,
                optimizer,
                model,
                data_loader,
                affinity_tune,
                fape_tune,
                torsion_tune,
                fine_tune,
                output_directory,
                animated_data,
                record,
            )

        # Create the metrics row for this epoch
        record.save()

    def _validate(
        self,
        epoch_index: int,
        model: Predictor,
        data_loader: DataLoader,
        affinity_tune: bool,
        fape_tune: bool,
        torsion_tune: bool,
        fine_tune: bool,
        output_directory: str,
        structure_builders_count: int = 0,
        minimize_energy: bool | None = True,
    ) -> float:
        """Run an evaluation of the model, thus no backward propagation.

        The model parameters stay the same in this call.

        Args:
            epoch_index: the number of the epoch to save the metrics to
            model: the model that will be evaluated
            data_loader: data to insert into the model
            affinity_tune: whether to include affinity loss in the backward propagation
            fape_tune: whether to include fape loss in backward propagation
            torsion_tune: whether to include torsion loss in backward propagation
            fine_tune: whether to include fine tuning losses in backward propagation
            output_directory: where to store the results
            structure_builders_count: how many structure builders (pdb writers) to run simultaneously
            minimize_energy: whether to energy minimize the structures or not
        Returns:
            the mean total loss for all inserted data
        """
        # start with an empty metrics record
        record = MetricsRecord(epoch_index, data_loader.dataset.name, output_directory)

        # put model in evaluation mode
        model.eval()

        datapoint_count = 0
        sum_of_losses = 0.0

        with torch.no_grad():
            for batch_index, batch_data in enumerate(data_loader):
                # Transfer batch to device
                batch_data = {
                    k: v.to(self._device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in batch_data.items()
                }

                # make the model generate output
                batch_output = model(batch_data)

                # calculate the losses, for monitoring only
                batch_loss = get_loss(
                    self.config.model_type,
                    batch_output,
                    batch_data,
                    affinity_tune,
                    fape_tune,
                    torsion_tune,
                    fine_tune,
                )

                # count the number of loss values
                datapoint_count += batch_data["peptide_aatype"].shape[0]
                sum_of_losses += batch_loss["total"].item() * batch_data["peptide_aatype"].shape[0]

                if structure_builders_count > 0:
                    output_structures_to_directory(
                        structure_builders_count,
                        output_directory,
                        data_loader.dataset.name,
                        batch_data,
                        batch_output,
                        minimize_energy,
                    )

                # Save to metrics
                record.add_batch(batch_loss, batch_output, batch_data)

        # Create the metrics row for this epoch
        record.save()

        if datapoint_count > 0:
            return sum_of_losses / datapoint_count
        else:
            return 0.0

    def test(
        self,
        test_loaders: list[DataLoader],
        run_id: str,
        animated_complex_ids: list[str],
        model_path: str,
        structure_builders_count: int,
        minimize_energy: bool,
    ):
        """Call this function instead of train, when you just want to test the model.

        It saves the generated structures to hdf5 files.

        Args:
            test_loaders: test datasets to run the model on
            run_id: run directory, where the model file is stored
            animated_complex_ids: list of complex names, of which the structures should be saved
            model_path: pretrained model to use
            structure_builders_count: number of simultaneous structure builder processes
            minimize_energy: whether to do energy minimization on the structure
        """
        _log.info(f"testing {model_path} on {structure_builders_count} structure builders")

        # init model
        model: Predictor | DataParallel[Predictor] = Predictor(self.config)

        model_state = torch.load(model_path, map_location=self._device)
        model_state = remove_module_prefix(model_state)

        if self._device.type == "cuda" and torch.cuda.device_count() > 1:
            model = DataParallel(model)

        model.to(device=self._device)
        model.eval()

        # load the pretrained model
        model.load_state_dict(model_state)
        model.to(dtype=self._float_dtype)

        # make the snapshots
        if animated_complex_ids is not None:
            datasets = [loader.dataset for loader in test_loaders]
            animated_data = self._get_selection_data_batch(datasets, animated_complex_ids)
            self._snapshot("test", model, run_id, animated_data)

        for test_loader in test_loaders:
            # if we can generate affinity losses, we should
            affinity_tune = False
            if (
                self.config.model_type == ModelType.REGRESSION
                and "affinity" in test_loader.dataset[0]
            ):
                affinity_tune = True

            elif (
                self.config.model_type == ModelType.CLASSIFICATION
                and "class" in test_loader.dataset[0]
            ):
                affinity_tune = True

            # if we can generate structure losses, we should
            structure_tune = "peptide_all_atom_positions" in test_loader.dataset[0]

            # run the model to output results
            with Timer(
                f"test on {test_loader.dataset.name}, {len(test_loader.dataset)} data points"
            ):
                self._validate(
                    -1,
                    model,
                    test_loader,
                    affinity_tune,
                    structure_tune,
                    structure_tune,
                    structure_tune,
                    run_id,
                    structure_builders_count,
                    minimize_energy,
                )

    @staticmethod
    def _get_selection_data_batch(
        datasets: list[ProteinLoopDataset], names: list[str]
    ) -> dict[str, torch.Tensor]:
        """Searches for the requested entries in the datasets and collates them together in a batch.

        This needs to be done when animating structures.

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
        """Pathname to use for storing the model"""
        return f"{run_id}/best-predictor.pth"

    def train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loaders: list[DataLoader],
        epoch_counts: tuple[int, int],
        disable_ba_loss: bool,
        disable_struct_loss: bool,
        run_id: str,
        pretrained_model_path: str,
        animated_complex_ids: list[str],
        patience: int,
        enable_compile: bool = False,
        enable_profiling: bool = False,
        profile_wait: int = 3,
        profile_warmup: int = 2,
        profile_active: int = 1,
        profile_repeat: int = 1,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = True,
    ):
        """Call this method for training a model

        Args:
            train_loader: dataset for training
            valid_loader: dataset for validation, selecting the best model and deciding on early stopping
            test_loaders: datasets for testing on
            epoch_counts: number of epochs to run in phase 1,2
            disable_ba_loss: turns off the BA loss term
            disable_struct_loss: turns off the structural loss terms
            run_id: directory to store the resulting files
            pretrained_model_path: an optional pretrained model file to start from
            animated_complex_ids: names of complexes to animate during the run. their structure snapshots will be stored in an HDF5
            patience: early stopping patience
            enable_profiling: whether to enable PyTorch profiler during training
            profile_wait: profiler schedule wait steps
            profile_warmup: profiler schedule warmup steps
            profile_active: profiler schedule active steps
            profile_repeat: profiler schedule repeat count
            record_shapes: whether to record tensor shapes in profiler
            profile_memory: whether to enable memory profiling
            with_stack: whether to record stack traces in profiler
        """
        # Set up the model
        predictor = Predictor(self.config)
        model = predictor
        if self._device.type == "cuda" and torch.cuda.device_count() > 1:
            model = DataParallel(predictor)

        model.to(device=self._device)

        # continue on a pretrained model, if provided
        if pretrained_model_path is not None:
            model_state = torch.load(pretrained_model_path, map_location=self._device)
            model_state = remove_module_prefix(model_state)

            model.load_state_dict(model_state)

        model.to(dtype=self._float_dtype)

        # define model paths, for writing to
        model_path = f"{run_id}/best-predictor.pth"

        # Keep track of the lowest loss value seen during the run.
        lowest_loss = float("inf")

        # make the initial snapshots for animation, if we're animating something
        animated_data = None
        if animated_complex_ids is not None:
            datasets = [train_loader.dataset, valid_loader.dataset] + [
                test_loader.dataset for test_loader in test_loaders
            ]
            animated_data = self._get_selection_data_batch(datasets, animated_complex_ids)

            self._snapshot("0.0", model, run_id, animated_data)

        # create training phases, setting: max_epochs, fape_tune, torsion_tune, affinity_tune, fine_tune
        training_phases = [
            TrainingPhase(
                epoch_counts[0],
                1e-3,
                not disable_struct_loss,
                not disable_struct_loss,
                not disable_ba_loss,
                False,
            ),
            TrainingPhase(
                epoch_counts[1],
                1e-3,
                not disable_struct_loss,
                not disable_struct_loss,
                not disable_ba_loss,
                not disable_struct_loss,
            ),
        ]

        # skip phases with 0 epochs
        training_phases = [phase for phase in training_phases if phase.max_epoch_count > 0]

        # freeze/unfreeze parameters
        predictor.switch_affinity_grad(training_phases[0].affinity_tune)
        predictor.switch_structure_grad(
            training_phases[0].fape_tune
            or training_phases[0].torsion_tune
            or training_phases[0].fine_tune
        )

        # optimizer
        optimizer = Adam(model.parameters(), lr=training_phases[0].lr)

        # init early stop
        early_stop = EarlyStopper(patience=patience)

        _log.info(f"begin with training phase: {training_phases[0]}")

        # Compile the model
        if enable_compile:
            model.compile()

        # do the actual learning iteration
        total_epoch_count = 0
        _log.info(f"start training phase {training_phases[0]}")
        while True:
            # early stopping, if no more improvement or just end of epochs reached
            if early_stop.stops_early() or training_phases[0].end_reached():
                # move on to the next training phase
                training_phases = training_phases[1:]
                _log.info(f"end training phase at epoch {total_epoch_count}")

                # if no more training phases, end training
                if len(training_phases) == 0:
                    return

                # reload best model state
                model_state = torch.load(model_path, map_location=self._device)
                model_state = remove_module_prefix(model_state)

                model.load_state_dict(model_state)

                # reset early stopping variables
                early_stop = EarlyStopper(patience=patience)

                # reset lowest loss variable, to be sure that future best models get saved
                lowest_loss = float("inf")

                _log.info(
                    f"begin new training phase at epoch {total_epoch_count}: {training_phases[0]}"
                )

                # freeze/unfreeze parameters
                predictor.switch_affinity_grad(training_phases[0].affinity_tune)
                predictor.switch_structure_grad(
                    training_phases[0].fape_tune
                    or training_phases[0].torsion_tune
                    or training_phases[0].fine_tune
                )

                # adjust learning rate
                optimizer.param_groups[0]["lr"] = training_phases[0].lr

            # train during epoch
            with Timer(f"train epoch {total_epoch_count}") as t:
                self._epoch(
                    total_epoch_count,
                    optimizer,
                    model,
                    train_loader,
                    training_phases[0].affinity_tune,
                    training_phases[0].fape_tune,
                    training_phases[0].torsion_tune,
                    training_phases[0].fine_tune,
                    run_id,
                    animated_data,
                    enable_profiling,
                    profile_wait,
                    profile_warmup,
                    profile_active,
                    profile_repeat,
                    record_shapes,
                    profile_memory,
                    with_stack,
                )
                t.add_to_title(f"on {len(train_loader.dataset)} data points")

            # validate
            with Timer(f"valid epoch {total_epoch_count}") as t:
                valid_loss = self._validate(
                    total_epoch_count,
                    model,
                    valid_loader,
                    training_phases[0].affinity_tune,
                    training_phases[0].fape_tune,
                    training_phases[0].torsion_tune,
                    training_phases[0].fine_tune,
                    run_id,
                )
                t.add_to_title(f"on {len(valid_loader.dataset)} data points")

            # update from the validation loss
            early_stop.update(valid_loss)

            # If the validation loss improves, save the model.
            if valid_loss < lowest_loss:
                lowest_loss = valid_loss

                _log.info(f"saving model at epoch {total_epoch_count}")
                torch.save(model.state_dict(), model_path)

            # test
            for test_loader in test_loaders:
                with Timer(f"test epoch {total_epoch_count}") as t:
                    self._validate(
                        total_epoch_count,
                        model,
                        test_loader,
                        training_phases[0].affinity_tune,
                        training_phases[0].fape_tune,
                        training_phases[0].torsion_tune,
                        training_phases[0].fine_tune,
                        run_id,
                    )
                    t.add_to_title(f"on {len(test_loader.dataset)} data points")

            # count epochs
            training_phases[0].update()
            total_epoch_count += 1

    def get_data_loader(
        self,
        data_path: str,
        batch_size: int,
        shuffle: bool | None = True,
        entry_ids: list[str] | None = None,
        name: str | None = None,
    ) -> DataLoader:
        """Builds a data loader from a hdf5 dataset path.

        Args:
            data_path: HDF5 path to load
            batch_size: number of data points per batch, that the loader should output
            device: to load the batch data on
            shuffle: whether to shuffle the order of the data
            entry_ids: an optional list of datapoint names to use, if omitted load all data.
            name: a name to put on the dataset
        Returns:
            a data loader, providing access to the requested data
        """
        # create dataset as entrypoint to the data hdf5 file
        dataset = ProteinLoopDataset(
            data_path,
            torch.device("cpu"),  # Always create on CPU
            self._float_dtype,
            peptide_maxlen=self.config.peptide_maxlen,
            protein_maxlen=self.config.protein_maxlen,
            entry_names=entry_ids,
        )
        if name is not None:
            dataset.name = name

        # wrap the dataset in a data loader
        loader = DataLoader(
            dataset,
            collate_fn=ProteinLoopDataset.collate,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.workers_count,
            pin_memory=True,
            persistent_workers=True if self.workers_count > 0 else False,
            drop_last=True,
        )

        return loader


def read_ids_from(path: str) -> list[str]:
    """Reads the contents of a text file as a list of data point ids"""
    ids = []
    with open(path) as file_:
        for line in file_:
            ids += line.strip().split()
    return ids


def random_subdivision(ids: list[str], fraction: float) -> tuple[list[str], list[str]]:
    """Randomly divides a list of ids in two, given a fraction

    Args:
        ids: the list to divide
        fraction: the amount (0.0 - 1.0) for the list to move to the second return list
    Returns:
        two lists of ids, that together make the original input list
    """
    n = int(round(fraction * len(ids)))

    shuffled = copy(ids)
    random.shuffle(shuffled)

    return shuffled[:-n], shuffled[-n:]


def get_excluded(names_from: list[str], names_exclude: list[str]) -> list[str]:
    """Remove the names from one list from the other list

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
    else:
        # no name given, so make a random one
        run_id = str(uuid4())

    # check if there's already a model
    pretrained_model_path = args.pretrained_model
    if os.path.isdir(run_id):
        if pretrained_model_path is None:
            pretrained_model_path = Trainer.get_model_path(run_id)
            if not os.path.isfile(pretrained_model_path):
                pretrained_model_path = None
    else:
        # create a directory to store the results in
        os.mkdir(run_id)

    # apply debug settings, if chosen so
    log_level = logging.INFO
    if args.debug:
        log_level = logging.DEBUG
        torch.autograd.set_detect_anomaly(True)

    # If the user wants to log to stdout, set it.
    # Otherwise log to a file.
    if args.log_stdout:
        logging.basicConfig(
            stream=sys.stdout,
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    else:
        logging.basicConfig(
            filename=f"{run_id}/swiftmhc.log",
            filemode="a",
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    seed = random.getrandbits(64)
    torch.manual_seed(seed)
    _log.info(f"setting manual seed {seed}")

    # If cuda is available, use it.
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        _log.info(f"using {device_count} cuda devices")
        device = torch.device("cuda")
    else:
        _log.info("using cpu device")
        device = torch.device("cpu")

    # 16-bit memory consumption per floating point number
    float_dtype = torch.bfloat16

    # Make sure we can use multiple workers:
    _log.debug(f"using {args.workers} workers")
    torch.multiprocessing.set_start_method("spawn")

    # Init trainer object with given arguments
    trainer = Trainer(device, float_dtype, args.workers, model_type)

    if args.test_only:
        if pretrained_model_path is None:
            raise ValueError("testing requires a pretrained model")

        # We do a test, no training
        test_loaders = [
            trainer.get_data_loader(test_path, args.batch_size, shuffle=False)
            for test_path in args.data_path
        ]
        trainer.test(
            test_loaders,
            run_id,
            args.animate,
            pretrained_model_path,
            args.builders,
            args.with_energy_minimization,
        )

    else:  # train, validate, test
        if len(args.data_path) >= 3:
            # the sets are separate HDF5 files
            train_path, valid_path = args.data_path[:2]
            test_paths = args.data_path[2:]

            _log.debug(f"training on {train_path}")
            _log.debug(f"validating on {valid_path}")
            _log.debug(f"testing on {test_paths}")

            train_loader = trainer.get_data_loader(train_path, args.batch_size, shuffle=True)
            valid_loader = trainer.get_data_loader(valid_path, args.batch_size, shuffle=False)
            test_loaders = [
                trainer.get_data_loader(test_path, args.batch_size, shuffle=False)
                for test_path in test_paths
            ]

        elif len(args.data_path) == 2:
            # assume that the two files are train and validation set, no test set
            train_path, valid_path = args.data_path

            train_loader = trainer.get_data_loader(train_path, args.batch_size, shuffle=True)
            valid_loader = trainer.get_data_loader(valid_path, args.batch_size, shuffle=False)
            test_loaders = []

            _log.debug(f"training on {train_path}")
            _log.debug(f"validating on {valid_path}")

        else:
            # only one hdf5 file for train, validation and optionally test.

            if args.test_subset_path is not None:
                # If a test subset path was provided, take that indicated fraction of the HDF5 file.
                test_entry_names = read_ids_from(args.test_subset_path)
                train_valid_entry_names = get_excluded(
                    get_entry_names(args.data_path[0]), test_entry_names
                )
                train_entry_names, valid_entry_names = random_subdivision(
                    train_valid_entry_names, 0.1
                )
            else:
                # Otherwise, make only a train and validation set
                train_entry_names, valid_entry_names = random_subdivision(
                    get_entry_names(args.data_path[0]), 0.1
                )
                test_entry_names = []

            _log.debug(f"training, validating & testing on {args.data_path[0]} subsets")

            data_name = os.path.splitext(os.path.basename(args.data_path[0]))[0]

            train_loader = trainer.get_data_loader(
                args.data_path[0], args.batch_size, True, train_entry_names, f"{data_name}-train"
            )
            valid_loader = trainer.get_data_loader(
                args.data_path[0], args.batch_size, False, valid_entry_names, f"{data_name}-valid"
            )

            test_loaders = []
            if len(test_entry_names) > 0:
                test_loaders = [
                    trainer.get_data_loader(
                        args.data_path[0],
                        args.batch_size,
                        False,
                        test_entry_names,
                        f"{data_name}-test",
                    )
                ]

        # train with the composed datasets and user-provided settings
        trainer.train(
            train_loader,
            valid_loader,
            test_loaders,
            (args.phase_1_epoch_count, args.phase_2_epoch_count),
            args.disable_ba_loss,
            args.disable_struct_loss,
            run_id,
            pretrained_model_path,
            args.animate,
            args.patience,
            args.enable_compile,
            args.enable_profiling,
            args.profile_wait,
            args.profile_warmup,
            args.profile_active,
            args.profile_repeat,
            not args.no_profile_shapes,
            not args.no_profile_memory,
            not args.no_profile_stack,
        )
