from tempfile import mkstemp
import os
from numpy import pi

import torch

from tcrspec.preprocess import preprocess
from tcrspec.dataset import ProteinLoopDataset


def test_preprocess_BA_67447():

    table_path = "tests/data/test-ba.csv"
    models_path = "tests/data"

    hdf5_file, hdf5_path = mkstemp()
    os.close(hdf5_file)

    try:
        preprocess(table_path, models_path, hdf5_path)

        # Check that the preprocessed data is complete
        dataset = ProteinLoopDataset(hdf5_path, torch.device("cpu"))
        entry = dataset[0]

        # kd from the table
        assert entry.kd == 6441.2

        # sequence from the table
        assert entry.loop_sequence == "GVNNLEHGL"

        # Verify correct shapes for t and R.
        assert entry.loop_s.shape[0] == entry.loop_t.shape[0]
        assert entry.loop_t.shape[-1] == 3

        assert entry.loop_s.shape[0] == entry.loop_R.shape[0]
        assert entry.loop_R.shape[-2:] == (3, 3)

        # Verify correct shape for z.
        assert entry.z.shape[0] == entry.loop_s.shape[0]
        assert entry.z.shape[1] == entry.protein_s.shape[0]

        # Verify that the torsion angles for LEU are set.
        assert entry.loop_a.shape[0] == entry.loop_s.shape[0]
        assert entry.loop_a.shape[-1] == 7
        assert torch.all(entry.loop_a[4, :5] != 0.0)

        # Verify that the side chain positions for LEU are set.
        assert entry.loop_x.shape[0] == entry.loop_s.shape[0]
        assert entry.loop_x.shape[-2:] == (14, 3)
        assert torch.all(entry.loop_x[0, :4, :] != 0.0)

        # Verify that GLU symmetry is taken into account.
        delta = torch.abs(entry.loop_a[5, 5] - entry.loop_a_alt[5, 5])
        assert abs(delta - pi) < 0.01

        # Verify correct shapes for t and R.
        assert entry.protein_s.shape[0] == entry.protein_t.shape[0]
        assert entry.protein_t.shape[-1] == 3

        assert entry.protein_s.shape[0] == entry.protein_R.shape[0]
        assert entry.protein_R.shape[-2:] == (3, 3)

        protein_s, protein_t, protein_R, protein_mask, \
        loop_s, loop_t, loop_R, loop_a, loop_a_alt, loop_x, loop_x_alt, loop_mask, \
        z, kd = ProteinLoopDataset.collate([dataset[0], dataset[1]])
        assert torch.all(loop_mask == True)
        assert torch.all(protein_mask[:, :275] == True)
        assert torch.any(protein_mask[:, 275] == True)
        assert not torch.all(protein_mask[:, 275] == True)
    finally:
        os.remove(hdf5_path)

