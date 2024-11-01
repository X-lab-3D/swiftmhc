from tempfile import mkstemp, gettempdir
import os
from numpy import pi
from uuid import uuid4
from math import log

import torch

from swiftmhc.preprocess import (preprocess,
                                 _save_protein_data,
                                 _load_protein_data,
                                 _generate_structure_data,
                                 _find_model_as_bytes)
from swiftmhc.dataset import ProteinLoopDataset
from swiftmhc.metrics import get_sequence


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
            False,
        )

        # Check that the preprocessed data is complete
        dataset = ProteinLoopDataset(hdf5_path, torch.device("cpu"), torch.float32, 16, 200)
        assert len(dataset) > 0

        entry = dataset[0]
        assert entry is not None
        assert entry["ids"] == "BA-67447"

        # from the table
        assert "affinity" in entry, entry.keys()
        assert entry['affinity'] == (1.0 - log(6441.2) / log(50000))

        # sequence from the table
        assert get_sequence(entry['peptide_aatype'], entry['peptide_self_residues_mask']) == "GVNNLEHGL"

        # Verify correct shapes for s and x
        assert entry['peptide_sequence_onehot'].shape[0] == entry['peptide_all_atom_positions'].shape[0]
        assert entry['peptide_all_atom_positions'].shape[-1] == 3

        # Verify correct shape for z.
        assert entry['protein_proximities'].shape[0] == entry['protein_proximities'].shape[1]
    finally:
        os.remove(hdf5_path)


def test_protein_data_preserved():

    models_path = "tests/data"
    allele = "HLA-A*02:01"
    ref_pdb_path = "tests/data/BA-55224.pdb"
    self_mask_path = "tests/data/hlaa0201-gdomain.mask"
    cross_mask_path = "tests/data/hlaa0201-binding-groove.mask"

    with open(ref_pdb_path, 'rb') as pdb:
        model_bytes = pdb.read()

    protein_data, peptide_data = _generate_structure_data(
        model_bytes,
        ref_pdb_path,
        self_mask_path,
        cross_mask_path,
        allele,
        torch.device("cpu"),
    )

    assert "proximities" in protein_data

    tmp_hdf5_path = os.path.join(gettempdir(), f"{uuid4()}.hdf5")   

    try:
        _save_protein_data(tmp_hdf5_path, allele, protein_data)
        loaded_protein_data = _load_protein_data(tmp_hdf5_path, allele)
    finally:
        if os.path.isfile(tmp_hdf5_path):
            os.remove(tmp_hdf5_path)

    assert "proximities" in loaded_protein_data
