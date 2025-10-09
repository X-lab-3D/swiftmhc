import h5py
import torch
from openfold.np.residue_constants import restypes
from swiftmhc.dataset import ProteinLoopDataset


def test_getitem_with_structural_data():
    """Test __getitem__ method with structural data."""
    protein_maxlen = 200
    peptide_maxlen = 16

    dataset = ProteinLoopDataset(
        "tests/data/data.hdf5", torch.device("cpu"), torch.float32, peptide_maxlen, protein_maxlen
    )
    result = dataset[0]

    assert isinstance(result, dict)
    assert result["ids"] == "BA-99998"
    assert "affinity" in result
    assert "class" in result
    assert "affinity_lt" in result
    assert "affinity_gt" in result

    # ------- Verify tensor shapes ---------------------------------------------
    # Peptide base fields
    assert result["peptide_aatype"].shape == (peptide_maxlen,)
    assert result["peptide_atom14_alt_gt_positions"].shape == (peptide_maxlen, 14, 3)
    assert result["peptide_atom14_gt_exists"].shape == (peptide_maxlen, 14)
    assert result["peptide_atom14_gt_positions"].shape == (peptide_maxlen, 14, 3)
    assert result["peptide_backbone_rigid_tensor"].shape == (peptide_maxlen, 4, 4)
    assert result["peptide_blosum62"].shape == (peptide_maxlen, 32)
    assert result["peptide_residue_numbers"].shape == (peptide_maxlen,)
    assert result["peptide_sequence_onehot"].shape == (peptide_maxlen, 32)
    # Peptide mask fields
    assert result["peptide_self_residues_mask"].shape == (peptide_maxlen,)
    assert result["peptide_cross_residues_mask"].shape == (peptide_maxlen,)
    # Peptide-specific fields
    assert result["peptide_alt_torsion_angles_sin_cos"].shape == (peptide_maxlen, 7, 2)
    assert result["peptide_torsion_angles_mask"].shape == (peptide_maxlen, 7)
    assert result["peptide_torsion_angles_sin_cos"].shape == (peptide_maxlen, 7, 2)
    # Peptide generated fields
    assert result["peptide_residue_index"].shape == (peptide_maxlen,)

    # Protein base fields
    assert result["protein_aatype"].shape == (protein_maxlen,)
    assert result["protein_atom14_alt_gt_positions"].shape == (protein_maxlen, 14, 3)
    assert result["protein_atom14_gt_exists"].shape == (protein_maxlen, 14)
    assert result["protein_atom14_gt_positions"].shape == (protein_maxlen, 14, 3)
    assert result["protein_backbone_rigid_tensor"].shape == (protein_maxlen, 4, 4)
    assert result["protein_blosum62"].shape == (protein_maxlen, 32)
    assert result["protein_residue_numbers"].shape == (protein_maxlen,)
    assert result["protein_sequence_onehot"].shape == (protein_maxlen, 32)
    # Protein mask fields
    assert result["protein_self_residues_mask"].shape == (protein_maxlen,)
    assert result["protein_cross_residues_mask"].shape == (protein_maxlen,)
    # Protein generated fields
    assert result["protein_residue_index"].shape == (protein_maxlen,)
    assert result["protein_proximities"].shape == (protein_maxlen, protein_maxlen, 1)

    # ------- Verify tensor types ----------------------------------------------
    # Peptide base fields
    assert result["peptide_aatype"].dtype == torch.long
    assert result["peptide_atom14_alt_gt_positions"].dtype == torch.float32
    assert result["peptide_atom14_gt_exists"].dtype == torch.bool
    assert result["peptide_atom14_gt_positions"].dtype == torch.float32
    assert result["peptide_backbone_rigid_tensor"].dtype == torch.float32
    assert result["peptide_blosum62"].dtype == torch.float32
    assert result["peptide_residue_numbers"].dtype == torch.int32
    assert result["peptide_sequence_onehot"].dtype == torch.float32
    # Peptide mask fields
    assert result["peptide_self_residues_mask"].dtype == torch.bool
    assert result["peptide_cross_residues_mask"].dtype == torch.bool
    # Peptide-specific fields
    assert result["peptide_alt_torsion_angles_sin_cos"].dtype == torch.float32
    assert result["peptide_torsion_angles_mask"].dtype == torch.bool
    assert result["peptide_torsion_angles_sin_cos"].dtype == torch.float32
    # Peptide generated fields
    assert result["peptide_residue_index"].dtype == torch.long

    # Protein base fields
    assert result["protein_aatype"].dtype == torch.long
    assert result["protein_atom14_alt_gt_positions"].dtype == torch.float32
    assert result["protein_atom14_gt_exists"].dtype == torch.bool
    assert result["protein_atom14_gt_positions"].dtype == torch.float32
    assert result["protein_backbone_rigid_tensor"].dtype == torch.float32
    assert result["protein_blosum62"].dtype == torch.float32
    assert result["protein_residue_numbers"].dtype == torch.int32
    assert result["protein_sequence_onehot"].dtype == torch.float32
    # Protein mask fields
    assert result["protein_self_residues_mask"].dtype == torch.bool
    assert result["protein_cross_residues_mask"].dtype == torch.bool
    # Protein generated fields
    assert result["protein_residue_index"].dtype == torch.long
    assert result["protein_proximities"].dtype == torch.float32

    # -------- Verify tensor values against actual HDF5 data -------------------
    entry_name = "BA-99998"
    with h5py.File("tests/data/data.hdf5", "r") as hdf5_file:
        group = hdf5_file[entry_name]

        hdf5_affinity = group["affinity"][()]
        assert torch.isclose(result["affinity"], torch.tensor(hdf5_affinity), atol=1e-6)

        if "class" in group:
            hdf5_class = group["class"][()]
            assert result["class"] == hdf5_class

        if "affinity_lt_mask" in group:
            hdf5_affinity_lt = bool(group["affinity_lt_mask"][()])
            assert result["affinity_lt"] == hdf5_affinity_lt

        if "affinity_gt_mask" in group:
            hdf5_affinity_gt = bool(group["affinity_gt_mask"][()])
            assert result["affinity_gt"] == hdf5_affinity_gt

        # Get sequence lengths for validation
        hdf5_peptide_aatype = group["peptide"]["aatype"][:]
        peptide_length = len(hdf5_peptide_aatype)
        hdf5_protein_aatype = group["protein"]["aatype"][:]
        protein_length = len(hdf5_protein_aatype)

        # ------- Verify tensor values for peptide -----------------------------
        # PEPTIDE BASE FIELDS
        result_peptide_aatype = result["peptide_aatype"][:peptide_length]
        expected_peptide_aatype = torch.from_numpy(hdf5_peptide_aatype).long()
        assert torch.equal(result_peptide_aatype, expected_peptide_aatype)

        if "atom14_alt_gt_positions" in group["peptide"]:
            hdf5_peptide_alt_positions = group["peptide"]["atom14_alt_gt_positions"][:]
            result_peptide_alt_positions = result["peptide_atom14_alt_gt_positions"][
                :peptide_length
            ]
            expected_peptide_alt_positions = torch.from_numpy(hdf5_peptide_alt_positions).float()
            assert torch.allclose(
                result_peptide_alt_positions, expected_peptide_alt_positions, atol=1e-6
            )

        if "atom14_gt_exists" in group["peptide"]:
            hdf5_peptide_atom_exists = group["peptide"]["atom14_gt_exists"][:]
            result_peptide_atom_exists = result["peptide_atom14_gt_exists"][:peptide_length]
            expected_peptide_atom_exists = torch.from_numpy(hdf5_peptide_atom_exists).bool()
            assert torch.equal(result_peptide_atom_exists, expected_peptide_atom_exists)

        if "atom14_gt_positions" in group["peptide"]:
            hdf5_peptide_positions = group["peptide"]["atom14_gt_positions"][:]
            result_peptide_positions = result["peptide_atom14_gt_positions"][:peptide_length]
            expected_peptide_positions = torch.from_numpy(hdf5_peptide_positions).float()
            assert torch.allclose(result_peptide_positions, expected_peptide_positions, atol=1e-6)

        if "backbone_rigid_tensor" in group["peptide"]:
            hdf5_peptide_rigid = group["peptide"]["backbone_rigid_tensor"][:]
            result_peptide_rigid = result["peptide_backbone_rigid_tensor"][:peptide_length]
            expected_peptide_rigid = torch.from_numpy(hdf5_peptide_rigid).float()
            assert torch.allclose(result_peptide_rigid, expected_peptide_rigid, atol=1e-6)

        if "blosum62" in group["peptide"]:
            hdf5_peptide_blosum = group["peptide"]["blosum62"][:]
            result_peptide_blosum = result["peptide_blosum62"][
                :peptide_length, : hdf5_peptide_blosum.shape[1]
            ]
            expected_peptide_blosum = torch.from_numpy(hdf5_peptide_blosum).float()
            assert torch.allclose(result_peptide_blosum, expected_peptide_blosum, atol=1e-6)

        if "residue_numbers" in group["peptide"]:
            hdf5_peptide_residue_numbers = group["peptide"]["residue_numbers"][:]
            result_peptide_residue_numbers = result["peptide_residue_numbers"][:peptide_length]
            expected_peptide_residue_numbers = torch.from_numpy(hdf5_peptide_residue_numbers).int()
            assert torch.equal(result_peptide_residue_numbers, expected_peptide_residue_numbers)

        if "sequence_onehot" in group["peptide"]:
            hdf5_peptide_onehot = group["peptide"]["sequence_onehot"][:]
            result_peptide_onehot = result["peptide_sequence_onehot"][
                :peptide_length, : hdf5_peptide_onehot.shape[1]
            ]
            expected_peptide_onehot = torch.from_numpy(hdf5_peptide_onehot).float()
            assert torch.allclose(result_peptide_onehot, expected_peptide_onehot, atol=1e-6)

        # PEPTIDE MASK FIELDS
        # Note: peptide self_residues_mask and cross_residues_mask are generated fields, not stored in HDF5
        peptide_self_mask_padding = result["peptide_self_residues_mask"][:peptide_length]
        assert torch.all(peptide_self_mask_padding), (
            "Peptide self residues mask padding should be True"
        )

        peptide_cross_mask_padding = result["peptide_cross_residues_mask"][:peptide_length]
        assert torch.all(peptide_cross_mask_padding), (
            "Peptide cross residues mask padding should be True"
        )

        # PEPTIDE-SPECIFIC FIELDS
        if "alt_torsion_angles_sin_cos" in group["peptide"]:
            hdf5_peptide_alt_torsions = group["peptide"]["alt_torsion_angles_sin_cos"][:]
            result_peptide_alt_torsions = result["peptide_alt_torsion_angles_sin_cos"][
                :peptide_length
            ]
            expected_peptide_alt_torsions = torch.from_numpy(hdf5_peptide_alt_torsions).float()
            assert torch.allclose(
                result_peptide_alt_torsions, expected_peptide_alt_torsions, atol=1e-6
            )

        if "torsion_angles_mask" in group["peptide"]:
            hdf5_peptide_torsion_mask = group["peptide"]["torsion_angles_mask"][:]
            result_peptide_torsion_mask = result["peptide_torsion_angles_mask"][:peptide_length]
            expected_peptide_torsion_mask = torch.from_numpy(hdf5_peptide_torsion_mask).bool()
            assert torch.equal(result_peptide_torsion_mask, expected_peptide_torsion_mask)

        if "torsion_angles_sin_cos" in group["peptide"]:
            hdf5_peptide_torsions = group["peptide"]["torsion_angles_sin_cos"][:]
            result_peptide_torsions = result["peptide_torsion_angles_sin_cos"][:peptide_length]
            expected_peptide_torsions = torch.from_numpy(hdf5_peptide_torsions).float()
            assert torch.allclose(result_peptide_torsions, expected_peptide_torsions, atol=1e-6)

        # ------- Verify tensor values for protein  ----------------------------
        # PROTEIN BASE FIELDS
        result_protein_aatype = result["protein_aatype"][:protein_length]
        expected_protein_aatype = torch.from_numpy(hdf5_protein_aatype).long()
        assert torch.equal(result_protein_aatype, expected_protein_aatype)

        if "atom14_alt_gt_positions" in group["protein"]:
            hdf5_protein_alt_positions = group["protein"]["atom14_alt_gt_positions"][:]
            result_protein_alt_positions = result["protein_atom14_alt_gt_positions"][
                :protein_length
            ]
            expected_protein_alt_positions = torch.from_numpy(hdf5_protein_alt_positions).float()
            assert torch.allclose(
                result_protein_alt_positions, expected_protein_alt_positions, atol=1e-6
            )

        if "atom14_gt_exists" in group["protein"]:
            hdf5_protein_atom_exists = group["protein"]["atom14_gt_exists"][:]
            result_protein_atom_exists = result["protein_atom14_gt_exists"][:protein_length]
            expected_protein_atom_exists = torch.from_numpy(hdf5_protein_atom_exists).bool()
            assert torch.equal(result_protein_atom_exists, expected_protein_atom_exists)

        if "atom14_gt_positions" in group["protein"]:
            hdf5_protein_positions = group["protein"]["atom14_gt_positions"][:]
            result_protein_positions = result["protein_atom14_gt_positions"][:protein_length]
            expected_protein_positions = torch.from_numpy(hdf5_protein_positions).float()
            assert torch.allclose(result_protein_positions, expected_protein_positions, atol=1e-6)

        if "backbone_rigid_tensor" in group["protein"]:
            hdf5_protein_rigid = group["protein"]["backbone_rigid_tensor"][:]
            result_protein_rigid = result["protein_backbone_rigid_tensor"][:protein_length]
            expected_protein_rigid = torch.from_numpy(hdf5_protein_rigid).float()
            assert torch.allclose(result_protein_rigid, expected_protein_rigid, atol=1e-6)

        if "blosum62" in group["protein"]:
            hdf5_protein_blosum = group["protein"]["blosum62"][:]
            result_protein_blosum = result["protein_blosum62"][
                :protein_length, : hdf5_protein_blosum.shape[1]
            ]
            expected_protein_blosum = torch.from_numpy(hdf5_protein_blosum).float()
            assert torch.allclose(result_protein_blosum, expected_protein_blosum, atol=1e-6)

        if "residue_numbers" in group["protein"]:
            hdf5_protein_residue_numbers = group["protein"]["residue_numbers"][:]
            result_protein_residue_numbers = result["protein_residue_numbers"][:protein_length]
            expected_protein_residue_numbers = torch.from_numpy(hdf5_protein_residue_numbers).int()
            assert torch.equal(result_protein_residue_numbers, expected_protein_residue_numbers)

        if "sequence_onehot" in group["protein"]:
            hdf5_protein_onehot = group["protein"]["sequence_onehot"][:]
            result_protein_onehot = result["protein_sequence_onehot"][
                :protein_length, : hdf5_protein_onehot.shape[1]
            ]
            expected_protein_onehot = torch.from_numpy(hdf5_protein_onehot).float()
            assert torch.allclose(result_protein_onehot, expected_protein_onehot, atol=1e-6)

        # PROTEIN MASK FIELDS
        if "self_residues_mask" in group["protein"]:
            hdf5_protein_self_mask = group["protein"]["self_residues_mask"][:]
            result_protein_self_mask = result["protein_self_residues_mask"][:protein_length]
            expected_protein_self_mask = torch.from_numpy(hdf5_protein_self_mask).bool()
            assert torch.equal(result_protein_self_mask, expected_protein_self_mask)

        if "cross_residues_mask" in group["protein"]:
            hdf5_protein_cross_mask = group["protein"]["cross_residues_mask"][:]
            result_protein_cross_mask = result["protein_cross_residues_mask"][:protein_length]
            expected_protein_cross_mask = torch.from_numpy(hdf5_protein_cross_mask).bool()
            assert torch.equal(result_protein_cross_mask, expected_protein_cross_mask)

        # PROTEIN GENERATED FIELDS
        if "proximities" in group["protein"]:
            hdf5_proximities = group["protein"]["proximities"][:]
            result_proximities = result["protein_proximities"][:protein_length, :protein_length]
            expected_proximities = torch.from_numpy(hdf5_proximities).float()
            assert torch.allclose(result_proximities, expected_proximities, atol=1e-6)

        # SEQUENCE RECONSTRUCTION VALIDATION
        peptide_sequence = "".join([restypes[i] for i in hdf5_peptide_aatype])
        result_peptide_sequence = "".join(
            [restypes[i] for i in result["peptide_sequence_onehot"].nonzero(as_tuple=True)[1]]
        )
        assert peptide_sequence == result_peptide_sequence

        protein_sequence = "".join([restypes[i] for i in hdf5_protein_aatype])
        result_protein_sequence = "".join(
            [restypes[i] for i in result["protein_sequence_onehot"].nonzero(as_tuple=True)[1]]
        )
        assert protein_sequence == result_protein_sequence

        # ------- Verify padding correctness ----------------------------------------

        # Peptide padding checks
        if peptide_length < peptide_maxlen:
            # PEPTIDE BASE FIELDS PADDING
            padding_values = result["peptide_aatype"][peptide_length:]
            assert torch.all(padding_values == 0), "Peptide aatype padding should be zeros"

            peptide_alt_positions_padding = result["peptide_atom14_alt_gt_positions"][
                peptide_length:
            ]
            assert torch.all(peptide_alt_positions_padding == 0), (
                "Peptide alt atom positions padding should be zeros"
            )

            peptide_atom_exists_padding = result["peptide_atom14_gt_exists"][peptide_length:]
            assert torch.all(~peptide_atom_exists_padding), (
                "Peptide atom exists mask padding should be False"
            )

            peptide_positions_padding = result["peptide_atom14_gt_positions"][peptide_length:]
            assert torch.all(peptide_positions_padding == 0), (
                "Peptide atom positions padding should be zeros"
            )

            peptide_rigid_padding = result["peptide_backbone_rigid_tensor"][peptide_length:]
            assert torch.all(peptide_rigid_padding == 0), (
                "Peptide backbone rigid tensor padding should be zeros"
            )

            peptide_blosum_padding = result["peptide_blosum62"][peptide_length:]
            assert torch.all(peptide_blosum_padding == 0), (
                "Peptide BLOSUM62 padding should be zeros"
            )

            peptide_residue_numbers_padding = result["peptide_residue_numbers"][peptide_length:]
            assert torch.all(peptide_residue_numbers_padding == 0), (
                "Peptide residue numbers padding should be zeros"
            )

            peptide_onehot_padding = result["peptide_sequence_onehot"][peptide_length:]
            assert torch.all(peptide_onehot_padding == 0), (
                "Peptide sequence one-hot padding should be zeros"
            )

            # PEPTIDE MASK FIELDS PADDING
            peptide_self_mask_padding = result["peptide_self_residues_mask"][peptide_length:]
            peptide_cross_mask_padding = result["peptide_cross_residues_mask"][peptide_length:]
            assert torch.all(~peptide_self_mask_padding), (
                "Peptide self residues mask padding should be False"
            )
            assert torch.all(~peptide_cross_mask_padding), (
                "Peptide cross residues mask padding should be False"
            )

            # PEPTIDE-SPECIFIC FIELDS PADDING
            peptide_alt_torsions_padding = result["peptide_alt_torsion_angles_sin_cos"][
                peptide_length:
            ]
            assert torch.all(peptide_alt_torsions_padding == 0), (
                "Peptide alt torsion angles padding should be zeros"
            )

            peptide_torsion_mask_padding = result["peptide_torsion_angles_mask"][peptide_length:]
            assert torch.all(~peptide_torsion_mask_padding), (
                "Peptide torsion angles mask padding should be False"
            )

            peptide_torsions_padding = result["peptide_torsion_angles_sin_cos"][peptide_length:]
            assert torch.all(peptide_torsions_padding == 0), (
                "Peptide torsion angles padding should be zeros"
            )

            # PEPTIDE GENERATED FIELDS PADDING
            peptide_residue_index_padding = result["peptide_residue_index"][peptide_length:]
            assert torch.all(peptide_residue_index_padding == 0), (
                "Peptide residue index padding should be zeros"
            )

        # Protein padding checks
        if protein_length < protein_maxlen:
            padding_values = result["protein_aatype"][protein_length:]
            assert torch.all(padding_values == 0), "Protein aatype padding should be zeros"

            protein_alt_positions_padding = result["protein_atom14_alt_gt_positions"][
                protein_length:
            ]
            assert torch.all(protein_alt_positions_padding == 0), (
                "Protein alt atom positions padding should be zeros"
            )

            protein_atom_exists_padding = result["protein_atom14_gt_exists"][protein_length:]
            assert torch.all(~protein_atom_exists_padding), (
                "Protein atom exists mask padding should be False"
            )

            protein_positions_padding = result["protein_atom14_gt_positions"][protein_length:]
            assert torch.all(protein_positions_padding == 0), (
                "Protein atom positions padding should be zeros"
            )

            protein_rigid_padding = result["protein_backbone_rigid_tensor"][protein_length:]
            assert torch.all(protein_rigid_padding == 0), (
                "Protein backbone rigid tensor padding should be zeros"
            )

            protein_blosum_padding = result["protein_blosum62"][protein_length:]
            assert torch.all(protein_blosum_padding == 0), (
                "Protein BLOSUM62 padding should be zeros"
            )

            protein_residue_numbers_padding = result["protein_residue_numbers"][protein_length:]
            assert torch.all(protein_residue_numbers_padding == 0), (
                "Protein residue numbers padding should be zeros"
            )

            protein_onehot_padding = result["protein_sequence_onehot"][protein_length:]
            assert torch.all(protein_onehot_padding == 0), (
                "Protein sequence one-hot padding should be zeros"
            )

            # PROTEIN MASK FIELDS PADDING
            protein_self_mask_padding = result["protein_self_residues_mask"][protein_length:]
            protein_cross_mask_padding = result["protein_cross_residues_mask"][protein_length:]
            assert torch.all(~protein_self_mask_padding), (
                "Protein self residues mask padding should be False"
            )
            assert torch.all(~protein_cross_mask_padding), (
                "Protein cross residues mask padding should be False"
            )

            # PROTEIN GENERATED FIELDS PADDING
            protein_residue_index_padding = result["protein_residue_index"][protein_length:]
            assert torch.all(protein_residue_index_padding == 0), (
                "Protein residue index padding should be zeros"
            )

            protein_proximities_row_padding = result["protein_proximities"][protein_length:, :]
            protein_proximities_col_padding = result["protein_proximities"][:, protein_length:]
            assert torch.all(protein_proximities_row_padding == 0), (
                "Protein proximities row padding should be zeros"
            )
            assert torch.all(protein_proximities_col_padding == 0), (
                "Protein proximities column padding should be zeros"
            )


def test_getitem_with_pairs():
    """Test __getitem__ method with peptide-allele pairs (prediction mode)."""
    protein_maxlen = 200
    peptide_maxlen = 16

    pairs = [("YLLGDSDSVA", "HLA-A*02:02")]
    dataset = ProteinLoopDataset(
        "tests/data/data.hdf5",
        torch.device("cpu"),
        torch.float32,
        peptide_maxlen,
        protein_maxlen,
        pairs=pairs,
    )
    result = dataset[0]

    assert isinstance(result, dict)
    assert "peptide" in result
    assert "allele" in result
    assert "ids" in result
    assert result["peptide"] == "YLLGDSDSVA"
    assert result["allele"] == "HLA-A*02:02"
    assert result["ids"] == "HLA-A*02:02-YLLGDSDSVA"
    assert "peptide_aatype" in result
    assert "peptide_sequence_onehot" in result
    assert "protein_aatype" in result
    assert "protein_sequence_onehot" in result


def test_collate():
    """Test collate method."""
    dataset = ProteinLoopDataset(
        "tests/data/data.hdf5", torch.device("cpu"), torch.float32, 16, 200
    )

    # Get the same entry twice to test batching
    i = dataset.entry_names.index("BA-99998")
    entry1 = dataset[i]
    entry2 = dataset[i]
    result = ProteinLoopDataset.collate([entry1, entry2])

    assert isinstance(result, dict)

    for key, value in entry1.items():
        if isinstance(value, torch.Tensor):
            assert key in result
            assert result[key].shape[0] == 2  # Batch size
            assert torch.equal(result[key][0], entry1[key])
            assert torch.equal(result[key][1], entry2[key])
        else:
            # Non-tensor fields should be in list format
            assert key in result
            assert isinstance(result[key], list)
            assert len(result[key]) == 2
            assert result[key][0] == entry1[key]
            assert result[key][1] == entry2[key]
