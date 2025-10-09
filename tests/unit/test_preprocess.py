import os
from math import log
from tempfile import gettempdir
from tempfile import mkstemp
from uuid import uuid4
import pytest
import torch
from Bio.PDB.Chain import Chain
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Residue import Residue
from swiftmhc.dataset import ProteinLoopDataset
from swiftmhc.domain.amino_acid import amino_acids_by_code
from swiftmhc.metrics import get_sequence
from swiftmhc.preprocess import _generate_structure_data
from swiftmhc.preprocess import _get_masked_structure
from swiftmhc.preprocess import _load_protein_data
from swiftmhc.preprocess import _read_mask_data
from swiftmhc.preprocess import _save_protein_data
from swiftmhc.preprocess import get_blosum_encoding
from swiftmhc.preprocess import preprocess


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
        assert entry["affinity"] == (1.0 - log(6441.2) / log(50000))

        # sequence from the table
        assert (
            get_sequence(entry["peptide_aatype"], entry["peptide_self_residues_mask"])
            == "GVNNLEHGL"
        )

        # Verify correct shapes for s and x
        assert (
            entry["peptide_sequence_onehot"].shape[0]
            == entry["peptide_all_atom_positions"].shape[0]
        )
        assert entry["peptide_all_atom_positions"].shape[-1] == 3

        # Verify correct shape for z.
        assert entry["protein_proximities"].shape[0] == entry["protein_proximities"].shape[1]
    finally:
        os.remove(hdf5_path)


def test_protein_data_preserved():
    models_path = "tests/data"
    allele = "HLA-A*02:01"
    ref_pdb_path = "tests/data/BA-55224.pdb"
    self_mask_path = "tests/data/hlaa0201-gdomain.mask"
    cross_mask_path = "tests/data/hlaa0201-binding-groove.mask"

    with open(ref_pdb_path, "rb") as pdb:
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

    xray_bytes = open(xray_path, "rb").read()
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
        amino_acid = amino_acids_by_code[residue_name]

        assert any(
            [r[0] == "M" and r[1] == residue_number and r[2] == amino_acid for r in self_mask]
        )


def test_get_blosum_encoding_basic_functionality():
    """Test basic functionality of get_blosum_encoding with list input."""
    device = torch.device("cpu")
    aa_indexes = [0, 1, 2, 3, 4]  # A, R, N, D, C

    result = get_blosum_encoding(aa_indexes, 62, device)

    assert result.shape == (5, 20), f"Expected shape (5, 20), got {result.shape}"
    assert result.device == device, f"Expected device {device}, got {result.device}"
    assert result.dtype == torch.float32, f"Expected float32, got {result.dtype}"

    # Check known BLOSUM62 values for Alanine (index 0)
    # A should have score 4 with itself (A is at index 0)
    assert result[0, 0] == 4.0, f"A-A score should be 4.0, got {result[0, 0]}"
    # A should have score -1 with R (R is at index 1)
    assert result[0, 1] == -1.0, f"A-R score should be -1.0, got {result[0, 1]}"


def test_get_blosum_encoding_invalid_amino_acid():
    """Test get_blosum_encoding with invalid amino acid indices."""
    device = torch.device("cpu")

    with pytest.raises(ValueError):
        invalid_indexes = []  # empty list
        get_blosum_encoding(invalid_indexes, 62, device)

    with pytest.raises(ValueError):
        invalid_indexes = [0, 1, 25]  # 25 is invalid
        get_blosum_encoding(invalid_indexes, 62, device)

    with pytest.raises(ValueError):
        invalid_indexes = [0, 1, -1]  # -1 is invalid
        get_blosum_encoding(invalid_indexes, 62, device)


def test_get_blosum_encoding_caching():
    """Test that caching works correctly and improves performance."""
    import time
    from swiftmhc.preprocess import _BLOSUM_CACHE

    device = torch.device("cpu")
    aa_indexes = list(range(20))

    # Clear cache to ensure clean test
    if 62 in _BLOSUM_CACHE:
        del _BLOSUM_CACHE[62]

    # First call should build cache
    start_time = time.time()
    result1 = get_blosum_encoding(aa_indexes, 62, device)
    first_call_time = time.time() - start_time

    # Verify cache was populated
    assert 62 in _BLOSUM_CACHE, "Cache should contain BLOSUM62 matrix after first call"
    assert _BLOSUM_CACHE[62].shape == (20, 20), "Cached matrix should be 20x20"

    # Second call should use cache and be faster
    start_time = time.time()
    result2 = get_blosum_encoding(aa_indexes, 62, device)
    second_call_time = time.time() - start_time

    assert result1.shape == (20, 20)
    assert torch.allclose(result1, result2), "Cached and non-cached results should be identical"
    assert second_call_time - first_call_time < 0, "Second call should complete successfully"
