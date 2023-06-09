import torch
from typing import Tuple
import h5py
from math import sqrt, log
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam
from scipy.stats import pearsonr


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

        mlp_input_size = 9 * 22
        c_affinity = 128

        self.aff_mlp = torch.nn.Sequential(
            torch.nn.Linear(mlp_input_size, c_affinity),
            torch.nn.GELU(),
            torch.nn.Linear(c_affinity, c_affinity),
            torch.nn.GELU(),
            torch.nn.Linear(c_affinity, 1), 
        )

    def forward(self, seq_embd: torch.Tensor) -> torch.Tensor:

        return self.aff_mlp(seq_embd.reshape(seq_embd.shape[0], -1)).reshape(seq_embd.shape[0])


if __name__ == "__main__":

    loss_func = torch.nn.MSELoss(reduction="mean")

    dataset = SequenceDataset("/data/tcrspec-clustered-10fold/train-fold2.hdf5")
    data_loader = DataLoader(dataset, batch_size=64)
    model = Model()
    model.train()

    optimizer = Adam(model.parameters(), lr=0.001)

    for epoch_index in range(10):

        for batch_input, affinity in data_loader:

            optimizer.zero_grad()

            output = model(batch_input.to(torch.float32))

            print("output", output)

            loss = loss_func(output, affinity.to(torch.float32))

            loss.backward()

            optimizer.step()

            print("loss", loss)
            print("pearson", _calc_pearson_correlation_coefficient(output, affinity))
            print("pearsonr", pearsonr(output.tolist(), affinity.tolist()))
