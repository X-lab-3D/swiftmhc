from math import pi, ceil, floor, log
import logging

import torch
import torch.nn
from torch.nn.functional import one_hot


_log = logging.getLogger(__name__)


def onehot_bin(x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    n_bins = bins.shape[0]

    b = torch.argmin(torch.abs(x.unsqueeze(-1).repeat([1] * len(x.shape) + [n_bins]) - bins), dim=-1)
    return one_hot(b, num_classes=n_bins)


def get_relative_position_encoding(sequence_mask: torch.Tensor, encoding_depth: int) -> torch.Tensor:
    """
    Args:
        sequence_mask: B x N (bool)

    D = encoding_depth

    Returns: a B x N x N x D tensor
    """

    # [B]
    sequence_lengths = sequence_mask.sum(dim=-1)

    bin_min = int(encoding_depth / 2) - 1
    bin_max = encoding_depth - bin_min
    bin_min = -bin_min

    # [D]
    bins = torch.arange(bin_min, bin_max, 1, device=sequence_mask.device)

    encs = []
    for index, length in enumerate(sequence_lengths):
        # [N]
        positions = torch.zeros(sequence_mask.shape[-1], device=sequence_mask.device, dtype=torch.long)
        positions[sequence_mask[index]] = torch.arange(0, length, 1, device=bins.device)

        # [N, N]
        d = positions.unsqueeze(-2) - positions.unsqueeze(-1)

        # [N, N]
        sqr_mask = sequence_mask[index].unsqueeze(-2) * sequence_mask[index].unsqueeze(-1)

        # [N, N, D]
        enc = onehot_bin(d, bins)
        enc[torch.logical_not(sqr_mask), :] = 0

        encs.append(enc)

    return torch.stack(encs).float()


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, max_len: int):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        self.d = d_model

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, seqs: torch.Tensor, masks: torch.Tensor):

        # each seq can have a different mask, that's why we have this iteration
        for i in range(seqs.shape[0]):
            length = masks[i].sum().item()
            seq = seqs[i]
            p = torch.zeros(length, seq.shape[-1], device=seq.device)
            p[:, :] = self.pe[:length, :]
            seq[masks[i]] += p
            seqs[i] = seq

        return seqs


class RelativePositionEncoding(torch.nn.Module):

    def __init__(self, c_z: int, relpos_k: int,):

        super(RelativePositionEncoding, self).__init__()

        self.relpos_k = relpos_k
        self.no_bins = 2 * relpos_k + 1
        self.linear_relpos = torch.nn.Linear(self.no_bins, c_z)

    def forward(self, ri: torch.Tensor) -> torch.Tensor:
        """
        Computes relative positional encodings

        Implements Algorithm 4.

        Args:
            ri:
                "residue_index" features of shape [*, N]
        """
        d = ri[..., None] - ri[..., None, :]
        boundaries = torch.arange(
            start=-self.relpos_k, end=self.relpos_k + 1, device=d.device
        )
        reshaped_bins = boundaries.view(((1,) * len(d.shape)) + (len(boundaries),))
        d = d[..., None] - reshaped_bins
        d = torch.abs(d)
        d = torch.argmin(d, dim=-1)
        d = torch.nn.functional.one_hot(d, num_classes=len(boundaries)).float()
        d = d.to(ri.dtype)

        return self.linear_relpos(d)


class RelativePositionEncodingWithOuterSum(torch.nn.Module):

    def __init__(self, c_z: int, relpos_k: int, tf_dim: int):

        super(RelativePositionEncodingWithOuterSum, self).__init__()

        self.linear_tf_z_i = torch.nn.Linear(tf_dim, c_z)
        self.linear_tf_z_j = torch.nn.Linear(tf_dim, c_z)

        self.relpos_k = relpos_k
        self.no_bins = 2 * relpos_k + 1
        self.linear_relpos = torch.nn.Linear(self.no_bins, c_z)

    def relpos(self, ri: torch.Tensor):

        d = ri[..., None] - ri[..., None, :]
        boundaries = torch.arange(
            start=-self.relpos_k, end=self.relpos_k + 1, device=d.device
        )
        reshaped_bins = boundaries.view(((1,) * len(d.shape)) + (len(boundaries),))
        d = d[..., None] - reshaped_bins
        d = torch.abs(d)
        d = torch.argmin(d, dim=-1)
        d = torch.nn.functional.one_hot(d, num_classes=len(boundaries)).float()
        d = d.to(ri.dtype)

        return self.linear_relpos(d)

    def forward(self, tf: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tf:
                "target_feat" features of shape [*, N_res, tf_dim]
        Returns:
            pair_emb:
                [*, N_res, N_res, C_z] pair embedding
        """

        ri = torch.arange(0, tf.shape[1], 1, dtype=torch.float, device=tf.device).unsqueeze(0).repeat(tf.shape[0], 1)

        relpos = self.relpos(ri)

        # [*, N_res, c_z]
        tf_emb_i = self.linear_tf_z_i(tf)
        tf_emb_j = self.linear_tf_z_j(tf)

        # [*, N_res, N_res, c_z]
        pair_emb = relpos + tf_emb_i[:, :, None, :] + tf_emb_j[:, None, :, :]

        return pair_emb

