import torch
from typing import Tuple
import h5py
from math import sqrt, log
from torch.utils.data import Dataset, DataLoader

from matplotlib import pyplot
from torch.optim import Adam
from scipy.stats import pearsonr

from tcrspec.modules.position_encoding import PositionalEncoding


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



class SequenceDataset(Dataset):

    def __init__(self, hdf5_path: str):

        self._hdf5_path = hdf5_path

        with h5py.File(self._hdf5_path, 'r') as hdf5_file:
            self._entry_names = list(hdf5_file.keys())

    def __len__(self) -> int:
        return len(self._entry_names)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:

        entry_name = self._entry_names[index]

        with h5py.File(self._hdf5_path, 'r') as hdf5_file:
            seq_embd = hdf5_file[entry_name]["loop/sequence_onehot"][:]
            affinity = 1.0 - log(hdf5_file[entry_name]["kd"][()]) / log(50000)

        return seq_embd, affinity


class Model(torch.nn.Module):

    def __init__(self):

        super(Model, self).__init__()

        mlp_input_size = 22
        c_res = 128
        c_affinity = 128

        self.pos_enc = PositionalEncoding(22, 9)

        self.res_mlp = torch.nn.Sequential(
            torch.nn.Linear(mlp_input_size, c_res),
            torch.nn.GELU(),
            torch.nn.Linear(c_res, c_res),
            torch.nn.GELU(),
            torch.nn.Linear(c_res, 1),
        )

        self.aff_mlp = torch.nn.Sequential(
            torch.nn.Linear(9, c_affinity),
            torch.nn.GELU(),
            torch.nn.Linear(c_affinity, c_affinity),
            torch.nn.GELU(),
            torch.nn.Linear(c_affinity, 1)
        )

        self.plot = False

    def forward(self, seq_embd: torch.Tensor) -> torch.Tensor:

        batch_size, loop_len, loop_depth = seq_embd.shape

        pos_enc = torch.eye(loop_len, loop_depth).unsqueeze(0).expand(seq_embd.shape)

        #seq_embd = torch.cat((seq_embd, pos_enc), dim=2)
        seq_embd = seq_embd + pos_enc

        if self.plot:
            figure = pyplot.figure()
            plot = figure.add_subplot()
            vmin=seq_embd.min().min()
            vmax=seq_embd.max().max()
            heatmap = plot.imshow(seq_embd[0], cmap="Greys", aspect="auto", vmin=vmin, vmax=vmax)
            figure.colorbar(heatmap)
            pyplot.show()

            self.plot = False

        prob = self.res_mlp(seq_embd)[..., 0]

        #prob = torch.nn.functional.softmax(prob, dim=1)

        #aff = torch.sum(-torch.log(prob), dim=1)

        aff = self.aff_mlp(prob)[..., 0]

        return aff


if __name__ == "__main__":

    loss_func = torch.nn.MSELoss(reduction="mean")

    train_dataset = SequenceDataset("/data/tcrspec-clustered-10fold/train-fold2.hdf5")
    train_data_loader = DataLoader(train_dataset, batch_size=64)

    test_dataset = SequenceDataset("/data/tcrspec-clustered-10fold/test-fold2.hdf5")
    test_data_loader = DataLoader(test_dataset, batch_size=64)

    model = Model()
    model.train()

    optimizer = Adam(model.parameters(), lr=0.001)

    for epoch_index in range(100):

        for batch_input, affinity in train_data_loader:

            optimizer.zero_grad()

            output = model(batch_input.to(torch.float32))

            loss = loss_func(output, affinity.to(torch.float32))

            loss.backward()

            optimizer.step()

        total_y = []
        total_z = []

        with torch.no_grad():
            for batch_input, affinity in test_data_loader:
                output = model(batch_input.to(torch.float32))

                total_y += output.tolist()
                total_z += affinity.tolist()

        print("pearsonr", pearsonr(total_y, total_z))
