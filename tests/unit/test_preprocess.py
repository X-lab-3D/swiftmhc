from tempfile import mkstemp
import os
from numpy import pi

import torch

from tcrspec.preprocess import preprocess
from tcrspec.dataset import ProteinLoopDataset
from tcrspec.metrics import get_sequence


def test_preprocess_BA_67447():

    table_path = "tests/data/test-ba.csv"
    models_path = "tests/data"
    self_mask_path = "tests/data/hlaa0201-gdomain.mask"
    cross_mask_path = "tests/data/hlaa0201-binding-groove.mask"
    ref_pdb_path = "tests/data/BA-55224.pdb"

    hdf5_file, hdf5_path = mkstemp()
    os.close(hdf5_file)
    os.remove(hdf5_path)

    try:
        preprocess(
            table_path,
            models_path,
            self_mask_path,
            cross_mask_path,
            hdf5_path,
            ref_pdb_path,
        )

        # Check that the preprocessed data is complete
        dataset = ProteinLoopDataset(hdf5_path, torch.device("cpu"), 16, 200)
        entry = dataset[0]

        # kd from the table
        assert entry['kd'] == 6441.2

        # sequence from the table
        assert get_sequence(entry['peptide_aatype'], entry['peptide_self_residues_mask']) == "GVNNLEHGL"

        # Verify correct shapes for s and x
        assert entry['peptide_sequence_onehot'].shape[0] == entry['peptide_all_atom_positions'].shape[0]
        assert entry['peptide_all_atom_positions'].shape[-1] == 3

        # Verify correct shape for z.
        assert entry['protein_proximities'].shape[0] == entry['protein_proximities'].shape[1]
    finally:
        os.remove(hdf5_path)

