import logging
import os
from typing import Any
import h5py
import numpy
import torch
from openfold.data.data_transforms import make_atom14_masks
from openfold.np import residue_constants
from torch.utils.data import Dataset
from .domain.amino_acid import AMINO_ACID_DIMENSION
from .domain.amino_acid import amino_acids_by_letter
from .preprocess import PREPROCESS_AFFINITY_GT_MASK_NAME
from .preprocess import PREPROCESS_AFFINITY_LT_MASK_NAME
from .preprocess import PREPROCESS_AFFINITY_NAME
from .preprocess import PREPROCESS_CLASS_NAME
from .preprocess import PREPROCESS_PEPTIDE_NAME
from .preprocess import PREPROCESS_PROTEIN_NAME
from .preprocess import affinity_binding_threshold
from .preprocess import get_blosum_encoding


_log = logging.getLogger(__name__)


def get_entry_names(hdf5_path: str) -> list[str]:
    """Get the list of entry names in the given HDF5 file."""
    try:
        with h5py.File(hdf5_path, "r") as hdf5_file:
            return list(hdf5_file.keys())
    except Exception as e:
        raise RuntimeError(f"Failed to open HDF5 file {hdf5_path}: {str(e)}")


class ProteinLoopDataset(Dataset):
    def __init__(
        self,
        hdf5_path: str,
        device: torch.device,
        float_dtype: torch.dtype,
        peptide_maxlen: int,
        protein_maxlen: int,
        entry_names: list[str] | None = None,
        pairs: list[tuple[str, str]] | None = None,
    ):
        """This class provides access to preprocessed data stored in HDF5 format.

        Args:
            hdf5_path: hdf5 file with structural data and optionally binding affinity
            device: cpu or cuda, must match with model
            peptide_maxlen: maximum length for storage of peptide data (in amino acids)
            protein_maxlen: maximum length for storage of protein data (in amino acids)
            entry_names: optional list of entries to use, by default all entries in the hdf5 are used
            pairs: optional list of pairs (peptide and allele) to combine, used for predicting unlabeled data with no structure
        """
        self.name = os.path.splitext(os.path.basename(hdf5_path))[0]

        self._hdf5_path = hdf5_path
        self._device = device
        self._peptide_maxlen = peptide_maxlen
        self._protein_maxlen = protein_maxlen

        if entry_names is not None:
            self._entry_names = entry_names
        else:
            # list all entries in the file
            self._entry_names = get_entry_names(self._hdf5_path)

        self._hdf5_file = None

        self._pairs = pairs

        self._float_dtype = float_dtype

    def _get_hdf5_file(self):
        """Get or create HDF5 file handle for this worker process.

        This method ensures thread-safe access to the HDF5 file by maintaining
        one file handle per worker process, avoiding the overhead of repeatedly
        opening and closing the file.
        """
        if self._hdf5_file is not None:
            return self._hdf5_file
        else:
            try:
                # "stdio" driver is faster for many small reads (<64KB)
                self._hdf5_file = h5py.File(self._hdf5_path, "r", driver="stdio")
            except Exception as e:
                raise RuntimeError(f"Failed to open HDF5 file {self._hdf5_path}: {str(e)}")
        return self._hdf5_file

    def _close_hdf5_file(self):
        """Close HDF5 file handle if open."""
        if self._hdf5_file is not None:
            try:
                self._hdf5_file.close()
            except Exception:
                pass  # Ignore errors during cleanup
            finally:
                self._hdf5_file = None

    def __del__(self):
        """Ensure HDF5 file is closed when dataset is destroyed."""
        self._close_hdf5_file()

    @property
    def entry_names(self) -> list[str]:
        """Get the list of entry names in this dataset."""
        return self._entry_names

    def __len__(self) -> int:
        if self._pairs is not None:
            # a set of requested MHC-peptide pairs, with unknown peptide structure or BA
            return len(self._pairs)
        else:
            # a set of preprocessed MHC-peptide structures
            return len(self._entry_names)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self._pairs is not None:
            # Pairs have been requested to be predicted, BA or peptide structure could be unknown.
            # We must make the combination.
            peptide, allele = self._pairs[index]

            try:
                entry_name = self._find_matching_entry(allele, peptide)
                result = self._get_structural_data(entry_name, False)
                result.update(self._get_sequence_data(peptide))
                result["peptide"] = peptide
                result["allele"] = allele
                result["ids"] = f"{allele}-{peptide}"

                return result
            except Exception as e:
                raise RuntimeError(f"for pair {peptide},{allele}: {str(e)}")
        else:
            # This is a preprocessed combination from the hdf5.
            entry_name = self._entry_names[index]

            try:
                return self.get_entry(entry_name)
            except Exception as e:
                raise RuntimeError(f"in entry {entry_name}: {str(e)}")

    def has_entry(self, entry_name: str) -> bool:
        """Whether the requested entry is in this dataset or not"""
        return entry_name in self._entry_names

    def _get_peptide_sequence(self, entry_name: str) -> str:
        """Gets the peptide sequence from the hdf5 file, under the given entry name"""
        hdf5_file = self._get_hdf5_file()
        aatype = hdf5_file[f"{entry_name}/{PREPROCESS_PEPTIDE_NAME}/aatype"][:]
        sequence = "".join(residue_constants.restypes[i] for i in aatype)

        return sequence

    def _find_matching_entry(self, allele_name: str, peptide_sequence: str | None = None) -> str:
        """Find an entry that matches the given allele and returns its name.

        The peptide sequence is optional and will only be matched when there are multiple entries with the matching allele.
        """
        if peptide_sequence is not None:
            peptide_aatype = [residue_constants.restypes.index(aa) for aa in list(peptide_sequence)]
        else:
            peptide_aatype = None

        matching_entry_name = None
        hdf5_file = self._get_hdf5_file()
        for entry_name, entry_group in hdf5_file.items():
            protein_group = entry_group[PREPROCESS_PROTEIN_NAME]

            if (
                "allele_name" in protein_group
                and protein_group["allele_name"][()].decode("utf_8") == allele_name
            ):
                # allele name match
                matching_entry_name = entry_name

                if peptide_aatype is not None and PREPROCESS_PEPTIDE_NAME in entry_group:
                    peptide_group = entry_group[PREPROCESS_PEPTIDE_NAME]
                    if numpy.all(peptide_group["aatype"] == peptide_aatype):
                        # perfect match
                        return entry_name

        if matching_entry_name is None:
            # not a single matching allele name
            raise ValueError(f"no entry found for {allele_name}")

        return matching_entry_name

    def _get_sequence_data(self, sequence: str) -> dict[str, torch.Tensor]:
        """Converts a peptide sequence into a SwiftMHC-compatible format

        Returns:
            a dictionary, containing all swiftMHC compatible data derived from the sequence
        """
        result = {}

        max_length = self._peptide_maxlen
        prefix = PREPROCESS_PEPTIDE_NAME
        length = len(sequence)
        amino_acids = [amino_acids_by_letter[a] for a in sequence]

        # amino acid mask
        mask = torch.zeros(self._peptide_maxlen, device=self._device, dtype=torch.bool)
        mask[:length] = True
        for interfix in ["self", "cross"]:
            result[f"{prefix}_{interfix}_residues_mask"] = mask

        # amino acid numbers
        aa_indices = [aa.index for aa in amino_acids]
        aatype_data = numpy.zeros(self._peptide_maxlen, dtype=numpy.int64)
        aatype_data[:length] = aa_indices
        result[f"{prefix}_aatype"] = (
            torch.from_numpy(aatype_data).to(device=self._device, dtype=torch.long).contiguous()
        )

        # one-hot encoding
        onehot_data = numpy.zeros((max_length, 32), dtype=numpy.float32)
        # Convert tensor to numpy if needed
        aa_onehot_list = []
        for aa in amino_acids:
            aa_onehot_list.append(aa.one_hot_code.cpu().numpy())
        aa_onehot = numpy.stack(aa_onehot_list)
        onehot_data[:length, :AMINO_ACID_DIMENSION] = aa_onehot
        result[f"{prefix}_sequence_onehot"] = (
            torch.from_numpy(onehot_data)
            .to(device=self._device, dtype=self._float_dtype)
            .contiguous()
        )

        # blosum62 encoding
        blosum_data = numpy.zeros((max_length, 32), dtype=numpy.float32)
        t = get_blosum_encoding([aa.index for aa in amino_acids], 62, device=torch.device("cpu"))
        blosum_data[:length, : t.shape[1]] = t.cpu().numpy()
        result[f"{prefix}_blosum62"] = (
            torch.from_numpy(blosum_data)
            .to(device=self._device, dtype=self._float_dtype)
            .contiguous()
        )

        # openfold needs each connected pair of residues to be one index apart
        residue_idx_data = numpy.zeros(max_length, dtype=numpy.int64)
        residue_idx_data[:length] = numpy.arange(0, length, 1)
        result[f"{prefix}_residue_index"] = (
            torch.from_numpy(residue_idx_data)
            .to(device=self._device, dtype=torch.long)
            .contiguous()
        )

        # residue numbers
        residue_num_data = numpy.zeros(max_length, dtype=numpy.int64)
        residue_num_data[:length] = numpy.arange(1, length + 1, 1)
        result[f"{prefix}_residue_numbers"] = (
            torch.from_numpy(residue_num_data)
            .to(device=self._device, dtype=torch.long)
            .contiguous()
        )

        # atoms masks, according to amino acids (vectorized: 10x faster!)
        atom14_mask_data = numpy.zeros((max_length, 14), dtype=bool)
        torsion_mask_data = numpy.zeros((max_length, 7), dtype=bool)
        all_atom_mask_data = numpy.zeros((max_length, 37), dtype=bool)

        aa_indices = [aa.index for aa in amino_acids]

        # Vectorized atom14 masks
        atom14_masks = numpy.array(
            [residue_constants.restype_atom14_mask[idx] for idx in aa_indices]
        )
        atom14_mask_data[:length] = atom14_masks

        # Vectorized torsion angle masks
        torsion_mask_data[:length, :3] = True  # First 3 are always True
        for i, aa_idx in enumerate(aa_indices):
            chi_mask = residue_constants.chi_angles_mask[aa_idx]
            torsion_mask_data[i, 3 : 3 + len(chi_mask)] = chi_mask

        # Vectorized all atom masks
        all_atom_masks = numpy.array(
            [residue_constants.restype_atom37_mask[idx] for idx in aa_indices]
        )
        all_atom_mask_data[:length] = all_atom_masks

        result[f"{prefix}_atom14_gt_exists"] = (
            torch.from_numpy(atom14_mask_data)
            .to(device=self._device, dtype=torch.bool)
            .contiguous()
        )
        result[f"{prefix}_torsion_angles_mask"] = (
            torch.from_numpy(torsion_mask_data)
            .to(device=self._device, dtype=torch.bool)
            .contiguous()
        )
        result[f"{prefix}_all_atom_mask"] = (
            torch.from_numpy(all_atom_mask_data)
            .to(device=self._device, dtype=torch.bool)
            .contiguous()
        )

        for key, value in make_atom14_masks({"aatype": result[f"{prefix}_aatype"]}).items():
            result[f"{prefix}_{key}"] = value

        return result

    def _read_hdf5_fields(self, group, field_names: list[str]) -> dict[str, Any]:
        """Read multiple HDF5 fields and return as a dictionary.

        Args:
            group: HDF5 group containing the datasets
            field_names: List of field names to read

        Returns:
            Dictionary mapping field names to their data arrays
        """
        data = {}
        valid_names = set(group.keys())
        for field_name in field_names:
            if field_name in valid_names:
                try:
                    data[field_name] = group[field_name][:]
                except Exception as e:
                    _log.warning(f"Failed to read field '{field_name}': {e}")
                    # Continue with other fields even if one fails
        return data

    def _get_required_fields_for_prefix(self, prefix: str, take_peptide: bool) -> list[str]:
        """Get the list of required HDF5 fields for a given prefix (protein/peptide)."""
        # Base fields needed for all prefixes
        base_fields = [
            "aatype",
            "atom14_alt_gt_positions",
            "atom14_gt_exists",
            "atom14_gt_positions",
            "backbone_rigid_tensor",
            "blosum62",
            "residue_numbers",
            "sequence_onehot",
        ]

        # Additional fields specific to peptide
        peptide_fields = [
            "alt_torsion_angles_sin_cos",
            "torsion_angles_mask",
            "torsion_angles_sin_cos",
        ]

        # Mask fields
        mask_fields = ["self_residues_mask", "cross_residues_mask"]

        required_fields = base_fields + mask_fields

        # Add peptide-specific fields if processing peptide
        if prefix == PREPROCESS_PEPTIDE_NAME and take_peptide:
            required_fields.extend(peptide_fields)

        return required_fields

    def _get_structural_data(self, entry_name: str, take_peptide: bool) -> dict[str, torch.Tensor]:
        """Takes structural data from the HDF5 file using optimized batch reading.

        Args:
            entry_name:     name of the entry (case) to which the structure belongs
            take_peptide:   whether or not to take the peptide structure also, if not then it takes only the MHC protein.

        Returns:
            a dictionary, containing all entry data taken from the HDF5.
        """
        result = {}

        hdf5_file = self._get_hdf5_file()
        entry_group = hdf5_file[entry_name]

        # Decide whether we take the peptide (if present) or just the protein residue data
        # In this iteration list:
        # 1. the group name: protein or peptide
        # 2. the maximum number of residues to fit, thus how much space to allocate
        # 3. the starting number in the list of indexes
        residue_iteration = [
            (PREPROCESS_PROTEIN_NAME, self._protein_maxlen, self._peptide_maxlen + 3)
        ]
        if take_peptide:
            residue_iteration.append((PREPROCESS_PEPTIDE_NAME, self._peptide_maxlen, 0))

        for prefix, max_length, start_index in residue_iteration:
            required_fields = self._get_required_fields_for_prefix(prefix, take_peptide)
            h5_data = self._read_hdf5_fields(entry_group[prefix], required_fields)

            # Validate aatype data (must exist)
            if "aatype" not in h5_data:
                raise ValueError(f"{entry_name} {prefix} missing required 'aatype' field")

            aatype_data = h5_data["aatype"]
            length = aatype_data.shape[0]
            if length < 3:
                raise ValueError(f"{entry_name} {prefix} length is {length}")

            elif length > max_length:
                raise ValueError(
                    f"{entry_name} {prefix} length is {length}, which is larger than the max {max_length}"
                )

            # Put all residues leftmost
            index = numpy.zeros(max_length, dtype=bool)
            index[:length] = True

            # Process aatype
            aatype_full = numpy.zeros(max_length, dtype=numpy.int64)
            aatype_full[index] = aatype_data
            result[f"{prefix}_aatype"] = (
                torch.from_numpy(aatype_full).to(device=self._device, dtype=torch.long).contiguous()
            )

            # Process mask fields using batch data
            for interfix in ["self", "cross"]:
                mask_full = numpy.zeros(max_length, dtype=bool)
                key = f"{interfix}_residues_mask"

                if key in h5_data:
                    mask_data = h5_data[key]
                    mask_full[index] = mask_data
                else:
                    # If no mask, then set all present residues to True.
                    mask_full[index] = True

                result[f"{prefix}_{interfix}_residues_mask"] = (
                    torch.from_numpy(mask_full)
                    .to(device=self._device, dtype=torch.bool)
                    .contiguous()
                )

            # openfold's loss functions need each connected pair of residues to be one index apart
            residue_idx_full = numpy.zeros(max_length, dtype=numpy.int64)
            residue_idx_full[index] = numpy.arange(start_index, start_index + length, 1)
            result[f"{prefix}_residue_index"] = (
                torch.from_numpy(residue_idx_full)
                .to(device=self._device, dtype=torch.long)
                .contiguous()
            )

            # identifiers of the residues within the chain: 1, 2, 3, ..
            residue_num_full = numpy.zeros(max_length, dtype=numpy.int32)
            if "residue_numbers" in h5_data:
                residue_num_full[index] = h5_data["residue_numbers"]
            result[f"{prefix}_residue_numbers"] = (
                torch.from_numpy(residue_num_full)
                .to(device=self._device, dtype=torch.int)
                .contiguous()
            )

            # one-hot encoded amino acid sequence
            onehot_full = numpy.zeros((max_length, 32), dtype=numpy.float32)
            if "sequence_onehot" in h5_data:
                seq_data = h5_data["sequence_onehot"]
                onehot_full[index, : seq_data.shape[1]] = seq_data
            result[f"{prefix}_sequence_onehot"] = (
                torch.from_numpy(onehot_full)
                .to(device=self._device, dtype=self._float_dtype)
                .contiguous()
            )

            # blosum 62 encoded amino acid sequence
            blosum_full = numpy.zeros((max_length, 32), dtype=numpy.float32)
            if "blosum62" in h5_data:
                blosum_data = h5_data["blosum62"]
                blosum_full[index, : blosum_data.shape[1]] = blosum_data
            result[f"{prefix}_blosum62"] = (
                torch.from_numpy(blosum_full)
                .to(device=self._device, dtype=self._float_dtype)
                .contiguous()
            )

            # backbone frames, used in IPA + atomic data, used in loss function
            structural_fields = [
                ("backbone_rigid_tensor", self._float_dtype),
                ("atom14_gt_positions", self._float_dtype),
                ("atom14_alt_gt_positions", self._float_dtype),
                ("atom14_gt_exists", torch.bool),
            ]
            # to save memory, don't add what we don't use.
            # only add torsion data for the peptide
            if prefix == PREPROCESS_PEPTIDE_NAME:
                structural_fields += [
                    ("torsion_angles_sin_cos", self._float_dtype),
                    ("alt_torsion_angles_sin_cos", self._float_dtype),
                    ("torsion_angles_mask", torch.bool),
                ]

            # Process structural fields using batch data
            for field_name, dtype in structural_fields:
                if field_name in h5_data:
                    data = h5_data[field_name]
                    # Create numpy array first
                    if dtype == torch.bool:
                        np_dtype = bool
                    elif dtype == self._float_dtype:
                        np_dtype = numpy.float32
                    else:
                        np_dtype = numpy.float32  # fallback

                    full_data = numpy.zeros([max_length] + list(data.shape[1:]), dtype=np_dtype)
                    full_data[index] = data
                    result[f"{prefix}_{field_name}"] = (
                        torch.from_numpy(full_data)
                        .to(device=self._device, dtype=dtype)
                        .contiguous()
                    )

        # protein residue-residue proximity data
        prox_data = entry_group[PREPROCESS_PROTEIN_NAME]["proximities"][:]
        prox_full = numpy.zeros(
            (self._protein_maxlen, self._protein_maxlen, 1), dtype=numpy.float32
        )
        prox_full[: prox_data.shape[0], : prox_data.shape[1], :] = prox_data
        result["protein_proximities"] = (
            torch.from_numpy(prox_full)
            .to(device=self._device, dtype=self._float_dtype)
            .contiguous()
        )

        return result

    def _set_zero_peptide_structure(
        self, result: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Make sure that for the peptide, all frames, atom positions and angles are set to zero.

        This is just to assure that the variables aren't missing during a run and
        can still be included in loss term calculation.
        This is meant for nonbinder structures where the loss term is calculated but doesn't count in the end.

        Args:
            result: a dictionary, to which variables are to be added.

        Returns:
            the resulting dictionary with the variables added.
        """
        max_length = self._peptide_maxlen

        # Important!
        # Must make sure that only ground truth values are set here.
        # Otherwise, the model can cheat.

        result["peptide_backbone_rigid_tensor"] = torch.zeros(
            (max_length, 4, 4), device=self._device, dtype=self._float_dtype
        )

        result["peptide_torsion_angles_sin_cos"] = torch.zeros(
            (max_length, 7, 2), device=self._device, dtype=self._float_dtype
        )
        result["peptide_alt_torsion_angles_sin_cos"] = torch.zeros(
            (max_length, 7, 2), device=self._device, dtype=self._float_dtype
        )

        result["peptide_atom14_gt_positions"] = torch.zeros(
            (max_length, 14, 3), device=self._device, dtype=self._float_dtype
        )
        result["peptide_atom14_alt_gt_positions"] = torch.zeros(
            (max_length, 14, 3), device=self._device, dtype=self._float_dtype
        )

        result["peptide_all_atom_positions"] = torch.zeros(
            (max_length, 37, 3), device=self._device, dtype=self._float_dtype
        )

        return result

    def get_entry(self, entry_name: str) -> dict[str, torch.Tensor]:
        """Gets the data entry (usually one pMHC case) with the given name(ID)

        Returns:
            a dictionary, containing all entry data taken from the HDF5.
        """
        result = {}

        hdf5_file = self._get_hdf5_file()
        entry_group = hdf5_file[entry_name]

        # store the id with the data entry
        result["ids"] = entry_name

        # The target affinity value is optional, thus only take it if present
        if PREPROCESS_AFFINITY_NAME in entry_group:
            result["affinity"] = torch.tensor(
                entry_group[PREPROCESS_AFFINITY_NAME][()],
                device=self._device,
                dtype=self._float_dtype,
            )

            if PREPROCESS_AFFINITY_LT_MASK_NAME in entry_group:
                result["affinity_lt"] = torch.tensor(
                    bool(entry_group[PREPROCESS_AFFINITY_LT_MASK_NAME][()]),
                    device=self._device,
                    dtype=torch.bool,
                )
            else:
                result["affinity_lt"] = torch.tensor(False, device=self._device, dtype=torch.bool)

            if PREPROCESS_AFFINITY_GT_MASK_NAME in entry_group:
                result["affinity_gt"] = torch.tensor(
                    bool(entry_group[PREPROCESS_AFFINITY_GT_MASK_NAME][()]),
                    device=self._device,
                    dtype=torch.bool,
                )
            else:
                result["affinity_gt"] = torch.tensor(False, device=self._device, dtype=torch.bool)

            result["class"] = result["affinity"] > affinity_binding_threshold

        if PREPROCESS_CLASS_NAME in entry_group:
            result["class"] = torch.tensor(
                entry_group[PREPROCESS_CLASS_NAME][()], device=self._device, dtype=torch.long
            )

        if "class" in result and result["class"] > 0:
            result.update(self._get_structural_data(entry_name, True))
        else:
            # nonbinders need no structural truth data for the peptide
            # only structural data for the protein
            result.update(self._get_structural_data(entry_name, False))

            peptide_sequence = self._get_peptide_sequence(entry_name)
            result.update(self._get_sequence_data(peptide_sequence))
            result = self._set_zero_peptide_structure(result)

        return result

    @staticmethod
    def collate(data_entries: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Collation function, to pack data of multiple entries in one batch.

        Returns:
            a dictionary, containing the batch data
        """
        if not data_entries:
            return {}

        if len(data_entries) == 1:
            return data_entries[0]

        # Get common keys more efficiently - start with first entry's keys
        # then filter out keys that don't exist in all entries
        common_keys = set(data_entries[0].keys())
        for entry in data_entries[1:]:
            common_keys &= entry.keys()

        # Pre-determine data types by checking the first entry only
        tensor_keys = set()
        list_keys = set()
        first_entry = data_entries[0]

        for key in common_keys:
            value = first_entry[key]
            if isinstance(value, torch.Tensor):
                tensor_keys.add(key)
            else:
                list_keys.add(key)

        result = {}

        # Batch process tensor keys (most common case)
        for key in tensor_keys:
            try:
                result[key] = torch.stack([entry[key] for entry in data_entries])
            except RuntimeError as e:
                raise RuntimeError(f"on {key}: {str(e)}")

        # Batch process list keys
        for key in list_keys:
            result[key] = [entry[key] for entry in data_entries]

        return result
