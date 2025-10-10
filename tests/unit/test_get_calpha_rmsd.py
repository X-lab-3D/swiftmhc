import pytest
import torch
from swiftmhc.loss import get_calpha_rmsd


class TestGetCalphaRmsd:
    """Test cases for the get_calpha_rmsd function."""

    def setup_method(self):
        """Set up common test data."""
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.batch_size = 4
        self.peptide_maxlen = 16

    def create_test_data(self, batch_size=None, include_class=True, include_affinity=False):
        """Create test data for get_calpha_rmsd function."""
        if batch_size is None:
            batch_size = self.batch_size

        output_data = {
            "final_positions": torch.randn(
                batch_size,
                self.peptide_maxlen,
                14,
                3,
                device=self.device,
                dtype=self.dtype,
            )
        }

        batch_data = {
            "peptide_atom14_gt_positions": torch.randn(
                batch_size,
                self.peptide_maxlen,
                14,
                3,
                device=self.device,
                dtype=self.dtype,
            ),
            "peptide_cross_residues_mask": torch.ones(
                batch_size, self.peptide_maxlen, device=self.device, dtype=torch.bool
            ),
            "ids": [f"entry_{i}" for i in range(batch_size)],
        }

        if include_class:
            # Create alternating pattern for larger batch sizes
            class_pattern = [1, 0] * (batch_size // 2) + [1] * (batch_size % 2)
            batch_data["class"] = torch.tensor(class_pattern, device=self.device, dtype=torch.long)

        if include_affinity:
            # Create alternating high/low affinity pattern for larger batch sizes
            affinity_pattern = [0.8, 0.3] * (batch_size // 2) + [0.8] * (batch_size % 2)
            batch_data["affinity"] = torch.tensor(
                affinity_pattern, device=self.device, dtype=self.dtype
            )

        return output_data, batch_data

    def test_basic_functionality_with_class(self):
        """Test basic RMSD calculation with classification data."""
        output_data, batch_data = self.create_test_data(include_class=True)
        result = get_calpha_rmsd(output_data, batch_data)
        # Only binders (entries 0 and 2) should be in the result
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result.values())
        assert all(v >= 0.0 for v in result.values())
        expected_ids = {"entry_0", "entry_2"}
        assert set(result.keys()) == expected_ids

    def test_basic_functionality_with_affinity(self):
        """Test basic RMSD calculation with affinity data."""
        output_data, batch_data = self.create_test_data(include_class=False, include_affinity=True)
        result = get_calpha_rmsd(output_data, batch_data)
        # Only high affinity entries (entries 0 and 2) should be in the result
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result.values())
        assert all(v >= 0.0 for v in result.values())

    def test_perfect_prediction_zero_rmsd(self):
        """Test that perfect predictions give zero RMSD."""
        output_data, batch_data = self.create_test_data(include_class=True)
        output_data["final_positions"] = batch_data["peptide_atom14_gt_positions"].clone()
        result = get_calpha_rmsd(output_data, batch_data)
        for rmsd in result.values():
            assert rmsd < 1e-6

    def test_masked_residues(self):
        """Test RMSD calculation with masked residues."""
        output_data, batch_data = self.create_test_data(include_class=True)
        batch_data["peptide_cross_residues_mask"][:, 5:] = False
        result = get_calpha_rmsd(output_data, batch_data)
        # Only binders (entries 0 and 2) should be in the result
        assert len(result) == 2
        assert all(v >= 0.0 for v in result.values())

    def test_empty_batch(self):
        """Test behavior with empty batch."""
        output_data = {"final_positions": torch.empty(0, self.peptide_maxlen, 14, 3)}
        batch_data = {
            "peptide_atom14_gt_positions": torch.empty(0, self.peptide_maxlen, 14, 3),
            "peptide_cross_residues_mask": torch.empty(0, self.peptide_maxlen, dtype=torch.bool),
            "class": torch.empty(0, dtype=torch.long),
            "ids": [],
        }
        result = get_calpha_rmsd(output_data, batch_data)
        assert result == {}

    def test_no_binders_batch(self):
        """Test behavior when batch has no binders."""
        output_data, batch_data = self.create_test_data(include_class=True)
        # Make all entries non-binders
        batch_data["class"] = torch.zeros(self.batch_size, dtype=torch.long)
        result = get_calpha_rmsd(output_data, batch_data)
        assert result == {}

    def test_single_entry_batch(self):
        """Test RMSD calculation with single entry."""
        output_data, batch_data = self.create_test_data(batch_size=1, include_class=True)
        result = get_calpha_rmsd(output_data, batch_data)
        assert len(result) == 1
        assert "entry_0" in result
        assert result["entry_0"] >= 0.0

    def test_missing_class_and_affinity_raises_error(self):
        """Test that missing both class and affinity data raises ValueError."""
        output_data, batch_data = self.create_test_data(include_class=False, include_affinity=False)
        with pytest.raises(
            ValueError, match="Cannot compute RMSD without class or affinity output"
        ):
            get_calpha_rmsd(output_data, batch_data)

    def test_rmsd_calculation_correctness(self):
        """Test that RMSD calculation is mathematically correct."""
        output_data, batch_data = self.create_test_data(batch_size=1, include_class=True)
        batch_data["peptide_atom14_gt_positions"][0, :, 1, :] = 0.0
        output_data["final_positions"][0, :3, 1, :] = torch.tensor([1.0, 0.0, 0.0])
        output_data["final_positions"][0, 3:, 1, :] = 0.0
        batch_data["peptide_cross_residues_mask"][0, 3:] = False
        result = get_calpha_rmsd(output_data, batch_data)
        expected_rmsd = 1.0
        assert abs(result["entry_0"] - expected_rmsd) < 1e-6

    def test_gpu_compatibility(self):
        """Test that function works with GPU tensors if CUDA is available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device("cuda")
        output_data, batch_data = self.create_test_data(include_class=True)
        output_data["final_positions"] = output_data["final_positions"].to(device)
        batch_data["peptide_atom14_gt_positions"] = batch_data["peptide_atom14_gt_positions"].to(
            device
        )
        batch_data["peptide_cross_residues_mask"] = batch_data["peptide_cross_residues_mask"].to(
            device
        )
        batch_data["class"] = batch_data["class"].to(device)
        result = get_calpha_rmsd(output_data, batch_data)
        assert len(result) == 2  # Only binders
        assert all(isinstance(v, float) for v in result.values())
