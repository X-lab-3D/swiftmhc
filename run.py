import os
import sys
from time import time
from argparse import ArgumentParser
import logging
from uuid import uuid4
from typing import Tuple, Union, Optional, List, Dict, Set, Any
import random
from math import log, sqrt
from multiprocessing import set_start_method
import h5py
import numpy
from io import StringIO

from scipy.stats import pearsonr
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
from tcrspec.loss import get_loss, get_calpha_square_deviation
from tcrspec.models.data import TensorDict
from tcrspec.tools.pdb import recreate_structure
from tcrspec.domain.amino_acid import amino_acids_by_one_hot_index


arg_parser = ArgumentParser(description="run a TCR-spec network model")
arg_parser.add_argument("--run-id", "-r", help="name of the run and the directory to store it")
arg_parser.add_argument("--debug", "-d", help="generate debug files", action='store_const', const=True, default=False)
arg_parser.add_argument("--log-stdout", "-l", help="log to stdout", action='store_const', const=True, default=False)
arg_parser.add_argument("--pretrained-model", "-m", help="use a given pretrained model state")
arg_parser.add_argument("--pretrained-protein-ipa", "-p", help="use a given pretrained protein ipa state")
arg_parser.add_argument("--test-only", "-t", help="skip training and test on a pretrained model", action='store_const', const=True, default=False)
arg_parser.add_argument("--batch-size", "-b", help="batch size to use during training/validation/testing", type=int, default=8)
arg_parser.add_argument("--epoch-count", "-e", help="how many epochs to run during training", type=int, default=100)
arg_parser.add_argument("--fine-tune-count", "-u", help="how many epochs to run during fine-tuning", type=int, default=10)
arg_parser.add_argument("--animate", "-a", help="id of a data point to generate intermediary pdb for")
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


class Trainer:
    def __init__(self,
                 device: torch.device):


        self._device = device

        self._early_stop_epsilon = 1e-6

        self._snap_period = 20

        self.loop_maxlen = 16
        self.protein_maxlen = 200

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

    def _snapshot(self,
                  frame_id: str,
                  model: Predictor,
                  output_directory: str,
                  data: Dict[str, torch.Tensor]):

        with torch.no_grad():
            output = model(data)

        id_ = data["ids"][0]
        animation_path = f"{output_directory}/{id_}-animation.hdf5"

        with h5py.File(animation_path, "a") as animation_file:

            frame_group = animation_file.require_group(frame_id)

            # save loop attentions heatmaps
            #loop_self_attention = output["loop_self_attention"].cpu()
            #frame_group.create_dataset("loop_attention", data=loop_self_attention[:, 0, ...], compression="lzf")

            # save loop embeddings heatmaps
            loop_embd = output["loop_embd"].cpu()
            frame_group.create_dataset("loop_embd", data=loop_embd[0, ...], compression="lzf")

            #loop_pos_enc = output["loop_pos_enc"].cpu()
            #frame_group.create_dataset("loop_pos_enc", data=loop_pos_enc[0, ...], compression="lzf")

            loop_init = output["loop_init"].cpu()
            frame_group.create_dataset("loop_init", data=loop_init[0, ...], compression="lzf")

            # save protein attentions heatmaps
            protein_self_attention = output["protein_self_attention"].cpu()
            protein_self_attention_sd = output["protein_self_attention_sd"].cpu()
            protein_self_attention_b = output["protein_self_attention_b"].cpu()
            frame_group.create_dataset("protein_attention", data=protein_self_attention[:, 0, ...], compression="lzf")
            frame_group.create_dataset("protein_attention_sd", data=protein_self_attention_sd[:, 0, ...], compression="lzf")
            frame_group.create_dataset("protein_attention_b", data=protein_self_attention_b[:, 0, ...], compression="lzf")

            # save cross attentions heatmaps
            cross_attention = output["cross_attention"].cpu()
            cross_attention_sd = output["cross_attention_sd"].cpu()
            cross_attention_pts = output["cross_attention_pts"].cpu()
            frame_group.create_dataset("cross_attention", data=cross_attention[:, 0, ...], compression="lzf")
            frame_group.create_dataset("cross_attention_sd", data=cross_attention_sd[:, 0, ...], compression="lzf")
            frame_group.create_dataset("cross_attention_pts", data=cross_attention_pts[:, 0, ...], compression="lzf")

            # save pdb
            structure = recreate_structure(id_,
                                           [("P", data["loop_residue_numbers"][0], data["loop_sequence_onehot"][0], output["final_positions"][0]),
                                            ("M", data["protein_residue_numbers"][0], data["protein_sequence_onehot"][0], data["protein_atom14_gt_positions"][0])])
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
            if not "protein_residue_numbers" in animation_file:
                animation_file.create_dataset("protein_residue_numbers", data=data["protein_residue_numbers"][0].cpu())

            if not "loop_residue_numbers" in animation_file:
                animation_file.create_dataset("loop_residue_numbers", data=data["loop_residue_numbers"][0].cpu())

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

        if "affinity" not in output_data:
            output_data["affinity"] = []
        output_data["affinity"] += truth["affinity"].tolist()

        if "output affinity" not in output_data:
            output_data["output affinity"] = []
        output_data["output affinity"] += output["affinity"].tolist()

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
            table = pandas.DataFrame(data={"id": [], "loop": [], "true affinity": [], "output affinity": []})
            table.set_index("id")

        for batch_index in range(batch_size):

            id_ = data["ids"][batch_index]

            loop_sequence = data["loop_sequence"][batch_index]

            row = pandas.DataFrame({"id": [id_], "loop": [loop_sequence],
                                    "output affinity": [data["output affinity"][batch_index]],
                                    "true affinity": [data["affinity"][batch_index]]})
            row.set_index("id")

            table = pandas.concat((table, row))

        table.to_csv(output_path, index=False)

    def test(self, test_loader: DataLoader, run_id: str):

        model_path = f"{run_id}/best-predictor.pth"
        model = Predictor(self.loop_maxlen, self.protein_maxlen,
                          openfold_config.model, self._device)
        model.to(device=self._device)
        model.eval()
        model.load_state_dict(torch.load(model_path))

        test_loss, test_data, test_output = self._validate(-1, model, test_loader, True, run_id)

        self._output_metrics(run_id, "test", -1, test_loss, test_data, test_output)

    @staticmethod
    def _get_single_data_batch(datasets: List[ProteinLoopDataset], name: str) -> Dict[str, torch.Tensor]:

        for dataset in datasets:
            if dataset.has_entry(name):
                return ProteinLoopDataset.collate([dataset.get_entry(name)])

        raise ValueError(f"entry no found in datasets: {name}")

    def train(self,
              train_loader: DataLoader,
              valid_loader: DataLoader,
              test_loader: DataLoader,
              epoch_count: int, fine_tune_count: int,
              run_id: Optional[str] = None,
              pretrained_model_path: Optional[str] = None,
              pretrained_protein_ipa_path: Optional[str] = None,
              animated_complex_id: Optional[str] = None):

        # get train data affinities
        train_affinities = torch.cat([batch_data["affinity"] for batch_data in train_loader])

        # Set up the model
        model = Predictor(self.loop_maxlen, self.protein_maxlen,
                          openfold_config.model)
        model.to(device=self._device)
        model.train()

        if pretrained_model_path is not None:
            model.load_state_dict(torch.load(pretrained_model_path,
                                             map_location=self._device))

        if pretrained_protein_ipa_path is not None:
            model.protein_ipa.load_state_dict(torch.load(pretrained_protein_ipa_path,
                                              map_location=self._device))

        optimizer = Adam(model.parameters(), lr=0.001)
        # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

        # define model paths
        model_path = f"{run_id}/best-predictor.pth"
        protein_ipa_path = f"{run_id}/best-protein-ipa.pth"

        # Keep track of the lowest loss value.
        lowest_loss = float("inf")

        animated_data = None
        if animated_complex_id is not None:
            # make snapshots for animation
            animated_data = self._get_single_data_batch([train_loader.dataset,
                                                         valid_loader.dataset,
                                                         test_loader.dataset], animated_complex_id)

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

            # write the metrics
            self._output_metrics(run_id, "train", epoch_index, train_data)
            self._output_metrics(run_id, "valid", epoch_index, valid_data)
            self._output_metrics(run_id, "test", epoch_index, test_data)

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
                torch.save(model.protein_ipa.state_dict(), protein_ipa_path)
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
                                                   "test total loss": [],
                                                   "train affinity correlation": [],
                                                   "valid affinity correlation": [],
                                                   "test affinity correlation": []})

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

        for loss_name in ("total loss", "affinity loss", "chi loss", "fape loss", "violation loss"):
            normalized_loss = data[loss_name] / len(data["ids"])

            metrics_dataframe.at[epoch_index, f"{pass_name} {loss_name}"] = round(normalized_loss, 3)

        try:
            pcc = pearsonr(data["output affinity"], data["affinity"]).statistic
            metrics_dataframe.at[epoch_index, f"{pass_name} affinity correlation"] = round(pcc, 3)
        except:
            output_aff = data["output affinity"]
            _log.exception(f"running pearsonr on {output_aff}")

        metrics_dataframe.at[epoch_index, f"{pass_name} binders C-alpha RMSD"] = round(data["binders_c_alpha_rmsd"], 3)

        metrics_dataframe.to_csv(metrics_path, sep=",", index=False)

    def get_data_loader(self,
                        data_path: str,
                        batch_size: int,
                        device: torch.device) -> DataLoader:

        dataset = ProteinLoopDataset(data_path, device, loop_maxlen=self.loop_maxlen, protein_maxlen=self.protein_maxlen)
        loader = DataLoader(dataset,
                            collate_fn=ProteinLoopDataset.collate,
                            batch_size=batch_size,
                            num_workers=5)

        return loader


if __name__ == "__main__":

    args = arg_parser.parse_args()

    if args.run_id is not None:
        run_id = args.run_id
    else:
        run_id = str(uuid4())

    if not os.path.isdir(run_id):
        os.mkdir(run_id)

    if args.log_stdout:
        logging.basicConfig(stream=sys.stdout,
                            level=logging.DEBUG if args.debug else logging.INFO)
    else:
        logging.basicConfig(filename=f"{run_id}/tcrspec.log", filemode="a",
                            level=logging.DEBUG if args.debug else logging.INFO)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        _log.debug("using cuda device")
    else:
        device = torch.device("cpu")
        _log.debug("using cpu device")

    set_start_method('spawn')

    trainer = Trainer(device)
    if args.test_only:

        test_path = args.data_path[0]

        test_loader = trainer.get_data_loader(test_path, args.batch_size, device)

        trainer.test(test_loader, run_id=run_id)

    else:  # train & test

        train_path, valid_path, test_path = args.data_path

        train_loader = trainer.get_data_loader(train_path, args.batch_size, device)
        valid_loader = trainer.get_data_loader(valid_path, args.batch_size, device)
        test_loader = trainer.get_data_loader(test_path, args.batch_size, device)

        trainer.train(train_loader, valid_loader, test_loader,
                      args.epoch_count, args.fine_tune_count,
                      run_id, args.pretrained_model, args.pretrained_protein_ipa, args.animate)
