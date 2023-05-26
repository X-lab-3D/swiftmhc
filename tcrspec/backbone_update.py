

import torch
import torch.nn


class BackboneUpdate(torch.nn.Module):
    def __init__(self, n_channels):

        super(BackboneUpdate, self).__init__()

        self._linear_b = torch.nn.Linear(n_channels, 1)
        self._linear_c = torch.nn.Linear(n_channels, 1)
        self._linear_d = torch.nn.Linear(n_channels, 1)
        self._linear_t = torch.nn.Linear(n_channels, 3)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: [n_sequences, n_residues, n_sequence_channels]
        Returns:
            [n_sequences, n_residues, 3, 4]
        """

        # [n_sequences, n_residues, 1]
        b = self._linear_b(s)

        # [n_sequences, n_residues, 1]
        c = self._linear_c(s)

        # [n_sequences, n_residues, 1]
        d = self._linear_d(s)

        # [n_sequences, n_residues, 3]
        t = self._linear_t(s)

        # [n_sequences, n_residues, 1]
        a_ones = torch.ones(*(b.shape))

        # [n_sequences, n_residues, 1]
        quaternion_norms = torch.sqrt(a_ones +
                                      torch.square(b) +
                                      torch.square(c) +
                                      torch.square(d))

        # [n_sequences, n_residues, 4]
        quaternions = torch.cat((a_ones, b, c, d), dim=2) / quaternion_norms

        # [n_sequences, n_residues, 1]
        a = quaternions[...,0]
        b = quaternions[...,1]
        c = quaternions[...,2]
        d = quaternions[...,3]

        # [n_sequences, n_residues, 3, 3]
        rotation_matrices = torch.stack((torch.stack((torch.square(a) + torch.square(b) - torch.square(c) - torch.square(d),
                                                      2 * b * c - 2 * a * d,
                                                      2 * b * d - 2 * a * c),
                                                     dim=2),
                                         torch.stack((2 * b * c + 2 * a * d,
                                                      torch.square(a) - torch.square(b) + torch.square(c) - torch.square(d),
                                                      2 * c * d - 2 * a * b),
                                                     dim=2),
                                         torch.stack((2 * b * d - 2 * a * c,
                                                      2 * c * d + 2 * a * b,
                                                      torch.square(a) - torch.square(b) - torch.square(c) + torch.square(d)),
                                                     dim=2)),
                                        dim=3)

        # [n_sequences, n_residues, 3, 1]
        t = t.view(list(t.shape) + [1])

        # [n_sequences, n_residues, 3, 4]
        return torch.cat((rotation_matrices, t), dim=3)
