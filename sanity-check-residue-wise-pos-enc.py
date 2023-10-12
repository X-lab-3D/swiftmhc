#!/usr/bin/env python

import torch
from typing import Tuple, Optional
import h5py
from math import sqrt, log
from torch.utils.data import Dataset, DataLoader

from matplotlib import pyplot
from torch.optim import Adam
from sklearn.metrics import matthews_corrcoef
from scipy.stats import pearsonr


from tcrspec.modules.position_encoding import PositionalEncoding


threshold = 1.0 - log(500) / log(50000)


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
            seq_embd = torch.zeros(9, 32)
            seq_embd[:, :22] = torch.tensor(hdf5_file[entry_name]["loop/sequence_onehot"][:])
            cls = hdf5_file[entry_name]["kd"][()] < 500.0

        return seq_embd, cls



class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self,
                 depth: int,
                 n_head: int,
                 dropout: Optional[float] = 0.1):

        super(TransformerEncoderLayer, self).__init__()

        self.n_head = n_head

        self.dropout = torch.nn.Dropout(dropout)

        self.linear_q = torch.nn.Linear(depth, depth * self.n_head, bias=False)
        self.linear_k = torch.nn.Linear(depth, depth * self.n_head, bias=False)
        self.linear_v = torch.nn.Linear(depth, depth * self.n_head, bias=False)

        self.linear_o = torch.nn.Linear(self.n_head * depth, depth, bias=False)

        self.norm_att = torch.nn.LayerNorm(depth)

        self.ff_intermediary_depth = 128 

        self.mlp_ff = torch.nn.Sequential(
            torch.nn.Linear(depth, self.ff_intermediary_depth),
            torch.nn.ReLU(),
            torch.nn.Linear(self.ff_intermediary_depth, depth),
        )

        self.norm_ff = torch.nn.LayerNorm(depth)

    def self_attention(
        self,
        seq: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, seq_len, d = seq.shape

        # [batch_size, n_head, seq_len, d]
        q = self.linear_q(seq).reshape(batch_size, seq_len, self.n_head, d).transpose(1, 2)
        k = self.linear_k(seq).reshape(batch_size, seq_len, self.n_head, d).transpose(1, 2)
        v = self.linear_v(seq).reshape(batch_size, seq_len, self.n_head, d).transpose(1, 2)

        # [batch_size, n_head, seq_len, seq_len]
        a = torch.softmax(torch.matmul(q, k.transpose(2, 3)) / sqrt(d), dim=3)

        # [batch_size, n_head, seq_len, d]
        heads = torch.matmul(a, v)

        # [batch_size, seq_len, d]
        o = self.linear_o(heads.transpose(1, 2).reshape(batch_size, seq_len, d * self.n_head))

        return o

    def feed_forward(self, seq: torch.Tensor) -> torch.Tensor:

        o = self.mlp_ff(seq)

        return o

    def forward(self,
                seq: torch.Tensor) -> torch.Tensor:

        x = seq

        x = self.dropout(x)

        y = self.self_attention(x)

        y = self.dropout(y)
        x = self.norm_att(x + y)

        y = self.feed_forward(x)

        y = self.dropout(y)
        x = self.norm_ff(x + y)

        return x


class Model(torch.nn.Module):

    def __init__(self):

        super(Model, self).__init__()

        mlp_input_size = 32
        c_res = 128
        c_affinity = 128

        self.pos_encoder = PositionalEncoding(32, 9)
        self.transf_encoder = TransformerEncoderLayer(32, 2)

        self.res_mlp = torch.nn.Sequential(
            torch.nn.Linear(mlp_input_size, c_res),
            torch.nn.GELU(),
            torch.nn.Linear(c_res, 1),
        )

    def forward(self, seq_embd: torch.Tensor) -> torch.Tensor:

        batch_size, loop_len, loop_depth = seq_embd.shape

        seq_embd = self.pos_encoder(seq_embd)
        seq_embd = self.transf_encoder(seq_embd)

        return self.res_mlp(seq_embd).sum(dim=2).sum(dim=1)


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
            for batch_input, true_cls in test_data_loader:
                output = model(batch_input.to(torch.float32))

                pred_cls = output >= threshold

                total_y += true_cls.tolist()
                total_z += pred_cls.tolist()

        corr = matthews_corrcoef(total_y, total_z)
        with open("results.csv", "at") as f:
            f.write(f"{corr}\n")
