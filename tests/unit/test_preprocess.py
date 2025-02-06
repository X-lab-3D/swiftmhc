from tempfile import mkstemp, gettempdir
import os
from numpy import pi
from uuid import uuid4
from math import log

import torch

from Bio.PDB.Chain import Chain
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Residue import Residue

from swiftmhc.preprocess import (preprocess,
                                 _get_masked_structure,
                                 _read_mask_data,
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


pdb_parser = PDBParser()


def _find_residue(chain: Chain, residue_number: int, residue_name: str) -> Residue:

    for residue in chain:
        if residue.get_id()[1] == residue_number and residue.get_resname() == residue_name:
            return residue

    raise ValueError(f"residue not found in {chain}: {residue_name} {residue_number}")


def test_alignment():

    xray_path = "tests/data/1AKJ.pdb"
    allele = "HLA-A*02:01"
    ref_path = "data/structures/reference-from-3MRD.pdb"
    self_mask_path = "data/HLA-A0201-GDOMAIN.mask"
    cross_mask_path = "data/HLA-A0201-CROSS.mask"

    self_mask = _read_mask_data(self_mask_path)
    cross_mask = _read_mask_data(cross_mask_path)

    xray_bytes = open(xray_path, 'rb').read()
    superposed_structure, masked_residue_dict = _get_masked_structure(
        xray_bytes,
        ref_path,
        {"self": self_mask, "cross": cross_mask},
        renumber_according_to_mask=True,
    )

    ref_structure = pdb_parser.get_structure("ref", ref_path)
    ref_model = list(ref_structure.get_models())[0]
    ref_chain_m = [chain for chain in ref_model if chain.get_id() == "M"][0]

    for residue, mask in masked_residue_dict["self"]:

        # verify that only chain M is aligned
        assert residue.get_parent().get_id() == "M"

        residue_number = residue.get_id()[1]
        residue_name = residue.get_resname()

        # verify that the residues have the same numbering:
        _find_residue(ref_chain_m, residue_number, residue_name)
