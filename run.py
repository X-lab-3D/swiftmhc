import os
import sys
from time import time
from argparse import ArgumentParser
import logging
from uuid import uuid4
from typing import Tuple, Union, Optional, List, Dict, Set, Any
import random
from math import log, sqrt
import h5py
import numpy
import shutil
from io import StringIO

from sklearn.metrics import matthews_corrcoef
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
from tcrspec.loss import get_loss, get_calpha_square_deviation
from tcrspec.models.data import TensorDict
from tcrspec.tools.pdb import recreate_structure
from tcrspec.domain.amino_acid import amino_acids_by_one_hot_index
from tcrspec.models.types import ModelType


arg_parser = ArgumentParser(description="run a TCR-spec network model")
arg_parser.add_argument("--run-id", "-r", help="name of the run and the directory to store it")
arg_parser.add_argument("--debug", "-d", help="generate debug files", action='store_const', const=True, default=False)
arg_parser.add_argument("--log-stdout", "-l", help="log to stdout", action='store_const', const=True, default=False)
arg_parser.add_argument("--pretrained-model", "-m", help="use a given pretrained model state")
arg_parser.add_argument("--pretrained-protein-ipa", "-p", help="use a given pretrained protein ipa state")
arg_parser.add_argument("--test-only", "-t", help="skip training and test on a pretrained model", action='store_const', const=True, default=False)
arg_parser.add_argument("--workers", "-w", help="number of workers to load batches", type=int, default=5)
arg_parser.add_argument("--batch-size", "-b", help="batch size to use during training/validation/testing", type=int, default=8)
arg_parser.add_argument("--epoch-count", "-e", help="how many epochs to run during training", type=int, default=100)
arg_parser.add_argument("--fine-tune-count", "-u", help="how many epochs to run during fine-tuning", type=int, default=10)
arg_parser.add_argument("--animate", "-a", help="id of a data point to generate intermediary pdb for", nargs="+")
arg_parser.add_argument("--structures-path", "-s", help="an additional structures hdf5 file to measure RMSD on")
arg_parser.add_argument("--classification", "-c", help="do classification instead of regression", action="store_const", const=True, default=False)
arg_parser.add_argument("data_path", help="path to the train, validation & test hdf5", nargs="+")


_log = logging.getLogger(__name__)


class Trainer:
    def __init__(self,
                 device: torch.device,
                 workers_count: int,
                 model_type: ModelType):

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
               fine_tune: bool,
               pdb_output_directory: Optional[str] = None,
               animated_complex_id: Optional[str] = None

    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        optimizer.zero_grad()

        output = model(data)

        losses = get_loss(output, data, fine_tune)

        loss = losses["total"]

        loss.backward()

        # only do this if necessary, when the training isn't stable
        #clip_grad_norm_(model.parameters(), 0.5)

        optimizer.step()

        return (losses.detach(), output)

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

    def _snapshot(self,
                  frame_id: str,
                  model: Predictor,
                  output_directory: str,
                  data: Dict[str, torch.Tensor]):

        with torch.no_grad():
            output = model(data)

        for index, id_ in enumerate(data["ids"]):

            animation_path = f"{output_directory}/{id_}-animation.hdf5"

            with h5py.File(animation_path, "a") as animation_file:

                # for convenience, store the true structure in the animation file also
                if "true" not in animation_file:
                    true_group = animation_file.require_group("true")
                    structure = recreate_structure(id_,
                                                   [("P", data["loop_residue_numbers"][index], data["loop_sequence_onehot"][index], data["loop_atom14_gt_positions"][index]),
                                                    ("M", data["protein_residue_numbers"][index], data["protein_sequence_onehot"][index], data["protein_atom14_gt_positions"][index])])
                    pdbio = PDBIO()
                    pdbio.set_structure(structure)
                    with StringIO() as sio:
                        pdbio.save(sio)
                        structure_data = numpy.array([bytes(line + "\n", encoding="utf-8")
                                                      for line in sio.getvalue().split('\n')
                                                      if len(line.strip()) > 0],
                                                     dtype=numpy.dtype("bytes"))
                    true_group.create_dataset("structure",
                                               data=structure_data,
                                               compression="lzf")

                frame_group = animation_file.require_group(frame_id)

                # save loop attentions heatmaps
                #loop_self_attention = output["loop_self_attention"].cpu()
                #frame_group.create_dataset("loop_attention", data=loop_self_attention[index], compression="lzf")

                # save loop embeddings heatmaps
                loop_embd = output["loop_embd"].cpu()
                frame_group.create_dataset("loop_embd", data=loop_embd[index], compression="lzf")

                #loop_pos_enc = output["loop_pos_enc"].cpu()
                #frame_group.create_dataset("loop_pos_enc", data=loop_pos_enc[index], compression="lzf")

                loop_init = output["loop_init"].cpu()
                frame_group.create_dataset("loop_init", data=loop_init[index], compression="lzf")

                # save protein attentions heatmaps
                protein_self_attention = output["protein_self_attention"].cpu()
                protein_self_attention_sd = output["protein_self_attention_sd"].cpu()
                protein_self_attention_b = output["protein_self_attention_b"].cpu()
                frame_group.create_dataset("protein_attention", data=protein_self_attention[index], compression="lzf")
                frame_group.create_dataset("protein_attention_sd", data=protein_self_attention_sd[index], compression="lzf")
                frame_group.create_dataset("protein_attention_b", data=protein_self_attention_b[index], compression="lzf")

                # save cross attentions heatmaps
                cross_attention = output["cross_attention"].cpu()
                cross_attention_sd = output["cross_attention_sd"].cpu()
                cross_attention_pts = output["cross_attention_pts"].cpu()
                frame_group.create_dataset("cross_attention", data=cross_attention[index], compression="lzf")
                frame_group.create_dataset("cross_attention_sd", data=cross_attention_sd[index], compression="lzf")
                frame_group.create_dataset("cross_attention_pts", data=cross_attention_pts[index], compression="lzf")

                # save pdb
                structure = recreate_structure(id_,
                                               [("P", data["loop_residue_numbers"][index], data["loop_sequence_onehot"][index], output["final_positions"][index]),
                                                ("M", data["protein_residue_numbers"][index], data["protein_sequence_onehot"][index], data["protein_atom14_gt_positions"][index])])
                pdbio = PDBIO()
                pdbio.set_structure(structure)
                with StringIO() as sio:
                    pdbio.save(sio)
                    structure_data = numpy.array([bytes(line + "\n", encoding="utf-8")
                                                  for line in sio.getvalue().split('\n')
                                                  if len(line.strip()) > 0],
                                                 dtype=numpy.dtype("bytes"))
                frame_group.create_dataset("structure",
                                           data=structure_data,
                                           compression="lzf")

                # save the residue numbering, for later lookup
                for key in ("protein_cross_residues_mask", "loop_cross_residues_mask",
                            "protein_residue_numbers", "loop_residue_numbers"):

                    if not key in animation_file:
                        animation_file.create_dataset(key, data=data[key][index].cpu())

    def _epoch(self,
               epoch_index: int,
               optimizer: Optimizer,
               model: Predictor,
               data_loader: DataLoader,
               fine_tune: bool,
               pdb_output_directory: Optional[str] = None,
               animated_data: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, Any]:

        epoch_data = {}

        model.train()

        sd = 0.0
        n = 0
        for batch_index, batch_data in enumerate(data_loader):

            # Do the training step.
            batch_loss, batch_output = self._batch(epoch_index,
                                                   batch_index, optimizer, model,
                                                   batch_data,
                                                   fine_tune)

            if pdb_output_directory is not None and animated_data is not None and (batch_index + 1) % self._snap_period == 0:

                self._snapshot(f"{epoch_index}.{batch_index + 1}",
                               model,
                               pdb_output_directory, animated_data)

            epoch_data = self._store_required_data(epoch_data, batch_loss, batch_output, batch_data)

            sum_, count = get_calpha_square_deviation(batch_output, batch_data)

            sd += sum_
            n += count

        epoch_data["binders_c_alpha_rmsd"] = sqrt(sd / n)

        return epoch_data

    def _validate(self,
                  epoch_index: int,
                  model: Predictor,
                  data_loader: DataLoader,
                  fine_tune: bool,
    ) -> Dict[str, Any]:

        valid_data = {}

        # using model.eval() here causes this issue:
        # https://github.com/pytorch/pytorch/pull/98375#issuecomment-1499504721

        sd = 0.0
        n = 0
        with torch.no_grad():

            for batch_index, batch_data in enumerate(data_loader):

                batch_size = batch_data["loop_sequence_onehot"].shape[0]

                batch_output = model(batch_data)

                batch_loss = get_loss(batch_output, batch_data, fine_tune)

                valid_data = self._store_required_data(valid_data, batch_loss, batch_output, batch_data)

                sum_, count = get_calpha_square_deviation(batch_output, batch_data)

                sd += sum_
                n += count

        valid_data["binders_c_alpha_rmsd"] = sqrt(sd / n)

        return valid_data

    @staticmethod
    def _store_required_data(old_data: Dict[str, Any],
                             losses: Dict[str, Any],
                             output: Dict[str, Any],
                             truth: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Stores the data needed to generate output files.
        """

        output_data = {}
        for key in old_data:
            output_data[key] = old_data[key]

        batch_size = len(truth["ids"])

        for loss_name, loss_value in losses.items():
            loss_name += " loss"

            if loss_name not in output_data:
                output_data[loss_name] = loss_value.item() * batch_size
            else:
                output_data[loss_name] += loss_value.item() * batch_size

        if "affinity" in truth:
            if "affinity" not in output_data:
                output_data["affinity"] = []
            output_data["affinity"] += truth["affinity"].tolist()

        if "class" in truth:
            if "class" not in output_data:
                output_data["class"] = []
            output_data["class"] += truth["class"].tolist()

        if "affinity" in output:
            if "output affinity" not in output_data:
                output_data["output affinity"] = []
            output_data["output affinity"] += output["affinity"].tolist()

        if "class" in output:
            if "output class" not in output_data:
                output_data["output class"] = []
            output_data["output class"] += output["class"].tolist()
            if "output classification" not in output_data:
                output_data["output classification"] = []
            output_data["output classification"] += output["classification"].tolist()

        if "ids" not in output_data:
            output_data["ids"] = []
        output_data["ids"] += truth["ids"]

        if "loop_sequence" not in output_data:
            output_data["loop_sequence"] = []
        output_data["loop_sequence"] += ["".join([aa.one_letter_code
                                                  for aa in one_hot_decode_sequence(embd)
                                                  if aa is not None])
                                         for embd in truth["loop_sequence_onehot"]]

        return output_data

    @staticmethod
    def _save_outputs_as_csv(output_path: str, data: Dict[str, Any]):

        batch_size = len(data["ids"])

        if os.path.isfile(output_path):
            table = pandas.read_csv(output_path)
        else:
            table = pandas.DataFrame(data={"id": [], "loop": []})
            if "affinity" in data:
                table["output affinity"] = []
                table["true affinity"] = []
            elif "class" in data:
                table["output class"] = []
                table["output 0"] = []
                table["output 1"] = []
                table["true class"] = []

            table.set_index("id")

        for batch_index in range(batch_size):

            id_ = data["ids"][batch_index]

            loop_sequence = data["loop_sequence"][batch_index]

            row_data = {"id": [id_], "loop": [loop_sequence]}
            if "affinity" in data and "output affinity" in data:
                row_data["output affinity"] = [data["output affinity"][batch_index]]
                row_data["true affinity"] = [data["affinity"][batch_index]]

            if "class" in data and "output class" in data:
                row_data["output class"] = [data["output class"][batch_index]]
                row_data["output 0"] = [data["output classification"][batch_index, 0]]
                row_data["output 1"] = [data["output classification"][batch_index, 1]]
                row_data["true class"] = [data["class"][batch_index]]

            row = pandas.DataFrame(row_data)
            row.set_index("id")

            table = pandas.concat((table, row))

        table.to_csv(output_path, index=False)

    def test(self, test_loader: DataLoader, structures_loader: Union[DataLoader, None], run_id: str):

        model_path = f"{run_id}/best-predictor.pth"
        model = Predictor(self._model_type,
                          self.loop_maxlen,
                          self.protein_maxlen,
                          openfold_config.model)
        model = DataParallel(model)

        model.to(device=self._device)
        model.eval()
        model.load_state_dict(torch.load(model_path))

        test_data = self._validate(-1, model, test_loader, True, run_id)

        structures_data = None
        if structures_loader is not None:
            structures_data = self._validate(-1, model, structures_loader, True, run_id)

        self._output_metrics(run_id, "test", -1, test_data)
        self._output_metrics(run_id, "structures", -1, structures_data)

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

    def train(self,
              train_loader: DataLoader,
              valid_loader: DataLoader,
              test_loader: DataLoader,
              epoch_count: int, fine_tune_count: int,
              run_id: Optional[str] = None,
              pretrained_model_path: Optional[str] = None,
              pretrained_protein_ipa_path: Optional[str] = None,

              animated_complex_ids: Optional[List[str]] = None,
              structures_loader: Optional[DataLoader] = None):

        # Set up the model
        model = Predictor(self._model_type,
                          self.loop_maxlen,
                          self.protein_maxlen,
                          openfold_config.model)
        model = DataParallel(model)

        model.to(device=self._device)
        model.train()

        if pretrained_model_path is not None:
            model.load_state_dict(torch.load(pretrained_model_path,
                                             map_location=self._device))

        if pretrained_protein_ipa_path is not None:
            model.module.protein_ipa.load_state_dict(torch.load(pretrained_protein_ipa_path,
                                                    map_location=self._device))

        optimizer = Adam(model.parameters(), lr=0.001)
        # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

        # define model paths
        model_path = f"{run_id}/best-predictor.pth"
        protein_ipa_path = f"{run_id}/best-protein-ipa.pth"

        # Keep track of the lowest loss value.
        lowest_loss = float("inf")

        animated_data = None
        if animated_complex_ids is not None:
            # make snapshots for animation
            datasets = [train_loader.dataset, valid_loader.dataset, test_loader.dataset]
            if structures_loader is not None:
                datasets.append(structures_loader.dataset)

            animated_data = self._get_selection_data_batch(datasets, animated_complex_ids)

            self._snapshot("0.0",
                           model,
                           run_id, animated_data)

        total_epochs = epoch_count + fine_tune_count
        for epoch_index in range(total_epochs):

            # flip this setting after the given number of epochs
            fine_tune = (epoch_index >= epoch_count)

            # train during epoch
            with Timer(f"train epoch {epoch_index}") as t:
                train_data = self._epoch(epoch_index, optimizer, model, train_loader, fine_tune,
                                         run_id, animated_data)
                t.add_to_title(f"on {len(train_loader.dataset)} data points")

            # validate
            with Timer(f"valid epoch {epoch_index}") as t:
                valid_data = self._validate(epoch_index, model, valid_loader, fine_tune)
                t.add_to_title(f"on {len(valid_loader.dataset)} data points")

            # test
            with Timer(f"test epoch {epoch_index}") as t:
                test_data = self._validate(epoch_index, model, test_loader, fine_tune)
                t.add_to_title(f"on {len(test_loader.dataset)} data points")

            # structures
            structures_data = None
            if structures_loader is not None:
                with Timer(f"structures epoch {epoch_index}") as t:
                    structures_data = self._validate(epoch_index, model, structures_loader, fine_tune)
                    t.add_to_title(f"on {len(structures_loader.dataset)} data points")

            # write the metrics
            self._output_metrics(run_id, "train", epoch_index, train_data)
            self._output_metrics(run_id, "valid", epoch_index, valid_data)
            self._output_metrics(run_id, "test", epoch_index, test_data)

            if structures_data is not None:
                self._output_metrics(run_id, "structures", epoch_index, structures_data)

            # early stopping, if no more improvement
            if abs(valid_data["total loss"] - lowest_loss) < self._early_stop_epsilon:
                if fine_tune:
                    break
                else:
                    epoch_index = epoch_count

            # If the loss improves, save the model.
            if valid_data["total loss"] < lowest_loss:
                lowest_loss = valid_data["total loss"]

                torch.save(model.state_dict(), model_path)
                torch.save(model.module.protein_ipa.state_dict(), protein_ipa_path)
            # else:
            #    model.load_state_dict(torch.load(model_path))

            # scheduler.step()

        # write the final test output
        self._save_outputs_as_csv(os.path.join(run_id, "outputs.csv"), test_data)

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

        metrics_path = f"{run_id}/metrics.csv"
        if os.path.isfile(metrics_path):

            metrics_dataframe = pandas.read_csv(metrics_path, sep=',')
        else:
            metrics_dataframe = self._init_metrics_dataframe()
        metrics_dataframe.set_index("epoch")

        while epoch_index > metrics_dataframe.shape[0]:
            metrics_dataframe = metrics_dataframe.append({name: [] for name in metrics_dataframe})

        metrics_dataframe.at[epoch_index, "epoch"] = int(epoch_index)

        for loss_name in filter(lambda s: s.endswith(" loss"), data.keys()):

            normalized_loss = data[loss_name] / len(data["ids"])

            metrics_dataframe.at[epoch_index, f"{pass_name} {loss_name}"] = round(normalized_loss, 3)

        if "output class" in data:
            try:
                mcc = matthews_corrcoef(data["output class"], data["class"])
                metrics_dataframe.at[epoch_index, f"{pass_name} matthews correlation"] = round(mcc, 3)
            except:
                output_class = data["output class"]
                _log.exception(f"running matthews_corrcoef on {output_class}")

        elif "output affinity" in data:
            try:
                pcc = pearsonr(data["output affinity"], data["affinity"]).statistic
                metrics_dataframe.at[epoch_index, f"{pass_name} affinity pearson correlation"] = round(pcc, 3)
            except:
                output_aff = data["output affinity"]
                _log.exception(f"running pearsonr on {output_aff}")

        metrics_dataframe.at[epoch_index, f"{pass_name} binders C-alpha RMSD"] = round(data["binders_c_alpha_rmsd"], 3)

        metrics_dataframe.to_csv(metrics_path, sep=",", index=False)

    def get_data_loader(self,
                        data_path: str,
                        batch_size: int,
                        device: torch.device) -> DataLoader:

        dataset = ProteinLoopDataset(data_path, device, self._model_type,
                                     loop_maxlen=self.loop_maxlen,
                                     protein_maxlen=self.protein_maxlen)

        loader = DataLoader(dataset,
                            collate_fn=ProteinLoopDataset.collate,
                            batch_size=batch_size,
                            num_workers=self.workers_count)

        return loader


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

    if args.log_stdout:
        logging.basicConfig(stream=sys.stdout,
                            level=logging.DEBUG if args.debug else logging.INFO)
    else:
        logging.basicConfig(filename=f"{run_id}/tcrspec.log", filemode="a",
                            level=logging.DEBUG if args.debug else logging.INFO)

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        _log.debug(f"using {device_count} cuda devices")

        device = torch.device("cuda")
    else:
        _log.debug("using cpu device")

        device = torch.device("cpu")

    _log.debug(f"using {args.workers} workers")
    torch.multiprocessing.set_start_method('spawn')

    trainer = Trainer(device, args.workers, model_type)

    structures_loader = None
    if args.structures_path is not None:
        structures_loader = trainer.get_data_loader(args.structures_path, args.batch_size, device)

    if args.test_only:

        test_path = args.data_path[0]

        test_loader = trainer.get_data_loader(test_path, args.batch_size, device)

        trainer.test(test_loader, structures_loader, run_id=run_id)

    else:  # train & test

        train_path, valid_path, test_path = args.data_path

        train_loader = trainer.get_data_loader(train_path, args.batch_size, device)
        valid_loader = trainer.get_data_loader(valid_path, args.batch_size, device)
        test_loader = trainer.get_data_loader(test_path, args.batch_size, device)

        trainer.train(train_loader, valid_loader, test_loader,
                      args.epoch_count, args.fine_tune_count,
                      run_id, args.pretrained_model, args.pretrained_protein_ipa,
                      args.animate, structures_loader)
