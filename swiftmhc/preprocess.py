from typing import List, Tuple, Union, Optional, Dict
import os
import logging
from math import isinf, floor, ceil, log
import tarfile
from uuid import uuid4
from tempfile import gettempdir

import h5py
import pandas
import numpy
import torch
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Structure import Structure
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.Align import PairwiseAligner
from Bio import SeqIO
from Bio.PDB.Polypeptide import is_aa, one_to_three
from blosum import BLOSUM

from pymol import cmd as pymol_cmd

from openfold.np.residue_constants import restype_atom37_mask, restype_atom14_mask, chi_angles_mask, restypes
from openfold.data.data_transforms import (atom37_to_frames,
                                           atom37_to_torsion_angles,
                                           get_backbone_frames,
                                           make_atom14_masks,
                                           make_atom14_positions)
from openfold.utils.feats import atom14_to_atom37

from .tools.pdb import get_atom14_positions
from .domain.amino_acid import amino_acids_by_letter, amino_acids_by_code, canonical_amino_acids
from .models.amino_acid import AminoAcid
from .models.complex import ComplexClass


_log = logging.getLogger(__name__)


# These strings represent the names under which data will be stored in the hdf5 file.
# They don't include everything, because some names are defined in the openfold code.
PREPROCESS_AFFINITY_LT_MASK_NAME = "affinity_lt_mask"
PREPROCESS_AFFINITY_GT_MASK_NAME = "affinity_gt_mask"
PREPROCESS_AFFINITY_NAME = "affinity"
PREPROCESS_CLASS_NAME = "class"
PREPROCESS_PROTEIN_NAME = "protein"
PREPROCESS_PEPTIDE_NAME = "peptide"


def _write_preprocessed_data(
    hdf5_path: str,
    storage_id: str,
    protein_data: Dict[str, torch.Tensor],
    peptide_data: Optional[Dict[str, torch.Tensor]] = None,
    affinity: Optional[float] = None,
    affinity_lt: Optional[bool] = False,
    affinity_gt: Optional[bool] = False,
    class_: Optional[ComplexClass] = None,
):
    """
    Output preprocessed protein-peptide data to and hdf5 file.

    Args:
        hdf5_path: path to output file
        storage_id: id to store the entry under as an hdf5 group
        protein_data: result output by '_read_residue_data' function, on protein residues
        peptide_data: result output by '_read_residue_data' function, on peptide residues
        affinity: the higher, the more tightly bound
        affinity_lt: a mask, true for <, false for =
        affinity_gt: a mask, true for >, false for =
        class_: BINDING/NONBINDING
    """

    _log.debug(f"writing {storage_id} to {hdf5_path}")

    with h5py.File(hdf5_path, 'a') as hdf5_file:

        storage_group = hdf5_file.require_group(storage_id)

        # store affinity/class data
        if affinity is not None:
            storage_group.create_dataset(PREPROCESS_AFFINITY_NAME, data=affinity)

        storage_group.create_dataset(PREPROCESS_AFFINITY_LT_MASK_NAME, data=affinity_lt)
        storage_group.create_dataset(PREPROCESS_AFFINITY_GT_MASK_NAME, data=affinity_gt)

        if class_ is not None:
            storage_group.create_dataset(PREPROCESS_CLASS_NAME, data=int(class_))

        # store protein data
        protein_group = storage_group.require_group(PREPROCESS_PROTEIN_NAME)
        for field_name, field_data in protein_data.items():
            if isinstance(field_data, torch.Tensor):
                protein_group.create_dataset(field_name, data=field_data.cpu(), compression="lzf")
            else:
                protein_group.create_dataset(field_name, data=field_data)

        # store peptide data
        if peptide_data is not None:
            peptide_group = storage_group.require_group(PREPROCESS_PEPTIDE_NAME)
            for field_name, field_data in peptide_data.items():
                if isinstance(field_data, torch.Tensor):
                    peptide_group.create_dataset(field_name, data=field_data.cpu(), compression="lzf")
                else:
                    peptide_group.create_dataset(field_name, data=field_data)


def _has_protein_data(
    hdf5_path: str,
    name: str,
) -> bool:
    """
    Check whether the preprocessed protein data is present.

    Args:
        hdf5_path: where it's stored
        name: the name in the hdf5, that it should be stored under
    """

    if not os.path.isfile(hdf5_path):
        return False

    with h5py.File(hdf5_path, 'r') as hdf5_file:
        return name in hdf5_file and PREPROCESS_PROTEIN_NAME in hdf5_file[name]

def _load_protein_data(
    hdf5_path: str,
    name: str,
) -> Dict[str, torch.Tensor]:
    """
    Load preprocessed protein data.

    Args:
        hdf5_path: where it's stored
        name: the name in the hdf5, that it's stored under
    Returns:
        the stored data
    """

    data = {}

    with h5py.File(hdf5_path, 'r') as hdf5_file:
        entry = hdf5_file[name]
        protein = entry[PREPROCESS_PROTEIN_NAME]

        for key in protein:
            _log.debug(f"{name}: loading {key} ..")

            value = protein[key][()]
            if isinstance(value, numpy.ndarray):

                data[key] = torch.tensor(value)
            else:
                data[key] = value

    return data


def _save_protein_data(
    hdf5_path: str,
    name: str,
    data: Dict[str, torch.Tensor]
):
    """
    Save preprocessed protein data.

    Args:
        hdf5_path: where to store it
        name: name to store it under in the hdf5
        data: to be stored
    """

    with h5py.File(hdf5_path, 'a') as hdf5_file:
        entry = hdf5_file.require_group(name)
        protein = entry.require_group(PREPROCESS_PROTEIN_NAME)

        for key in data:
            _log.debug(f"{name}: saving {key} ..")

            if isinstance(data[key], torch.Tensor):
                protein.create_dataset(key, data=data[key].cpu())
            else:
                protein.create_dataset(key, data=data[key])


# Representation of a line in the mask file:
# chain id, residue number, amino acid
ResidueMaskType = Tuple[str, int, AminoAcid]

def _read_mask_data(path: str) -> List[ResidueMaskType]:
    """
    Read from the mask TSV file, which residues in the PDB file should be marked as True.

    Format: CHAIN_ID  RESIDUE_NUMBER  AMINO_ACID_THREE_LETTER_CODE

    Lines starting in '#' will be ignored

    Args:
        path: input TSV file
    Returns:
        list of residues, present in the mask file
    """

    mask_data = []

    with open(path, 'r') as f:
        for line in f:
            if not line.startswith("#"):
                row = line.split()

                chain_id = row[0]
                residue_number = int(row[1])
                amino_acid = amino_acids_by_code[row[2]]

                mask_data.append((chain_id, residue_number, amino_acid))

    return mask_data


def _get_blosum_encoding(amino_acid_indexes: List[int], blosum_index: int, device: torch.device) -> torch.Tensor:
    """
    Convert amino acids to BLOSUM encoding

    Arguments:
        amino_acid_indexes: order of numbers 0 to 19, coding for the amino acids
        blosum_index: identifies the type of BLOSUM matrix to use
        device: to store the result on
    Returns:
        [len, 20] the amino acids encoded by their BLOSUM rows
    """

    matrix = BLOSUM(blosum_index)
    encoding = []
    for amino_acid_index in amino_acid_indexes:
        amino_acid = canonical_amino_acids[amino_acid_index]

        row = []
        for other_amino_acid in canonical_amino_acids:

            if isinf(matrix[amino_acid.one_letter_code][other_amino_acid.one_letter_code]):

                raise ValueError(f"not found in blosum matrix: {amino_acid.one_letter_code} & {other_amino_acid.one_letter_code}")
            else:
                row.append(matrix[amino_acid.one_letter_code][other_amino_acid.one_letter_code])

        encoding.append(row)

    return torch.tensor(encoding)


def _map_alignment(
    aln_path: str,
    aligned_structures: Tuple[Structure, Structure],
) -> List[Tuple[Union[Residue, None], Union[Residue, None]]]:
    """
    Maps a clustal alignment (.aln) file onto two structures

    Args:
        aln_path: clustal alignment (.aln) file
        aligned_structures: structures as in the order of the clustal alignment (.aln) file
    Returns:
        the aligned residues from the structures, as in the clustal alignment (.aln) file
    """

    # parse the alignment
    alignment = {}
    with open(aln_path) as handle:
        for record in SeqIO.parse(handle, "clustal"):
            alignment[record.id] = str(record.seq)

    # put the residues in the order of the alignment
    # assume the structures are presented in the same order
    maps = []
    for structure in aligned_structures:

        # skip structures that are not mentioned in the alignment file
        key = structure.get_id()
        if key not in alignment:
            continue

        # aligned sequence
        sequence = alignment[key]

        # residues in the structure, that must match with the aligned sequence
        residues = [r for r in structure.get_residues() if is_aa(r.get_resname())]

        _log.debug(f"mapping to {len(residues)} {key} residues:\n{sequence}")

        # match each letter in the aligned sequence with a residue in the structure
        map_ = []
        offset = 0
        for i in range(len(sequence)):
            if sequence[i].isalpha():

                # one letter amino acid code
                letter = sequence[i]

                # does the structure have more residues?
                if offset >= len(residues):
                    raise ValueError(f"{key} alignment has over {offset} residues, but the structure only has {len(residues)}")

                # match alignment code with amino acid
                if letter != 'X' and one_to_three(letter) != residues[offset].get_resname():
                    _log.warning(f"encountered {residues[offset].get_resname()} at {offset}, {one_to_three(letter)} expected")

                # store aligned structure residue
                map_.append(residues[offset])

                # go to next residue in the structure
                offset += 1
            else:
                map_.append(None)
        maps.append(map_)

    # zip the residues of the two structures
    results = [(maps[0][i], maps[1][i]) for i in range(len(maps[0]))]

    return results


def _make_sequence_data(sequence: str) -> Dict[str, torch.Tensor]:
    """
    Convert a sequence into a format that SwiftMHC can work with.

    Args:
        sequence: one letter codes of amino acids

    Returns:
        residue_numbers: [len] numbers of the residue as in the structure
        aatype: [len] sequence, indices of amino acids
        sequence_onehot: [len, 22] sequence, one-hot encoded amino acids
        blosum62: [len, 20] sequence, BLOSUM62 encoded amino acids
        torsion_angles_mask: [len, 7] which torsion angles each residue should have (openfold format)
        atom14_gt_exists: [len, 14] which atom each residue should have (openfold 14 format)
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    length = len(sequence)
    residue_numbers = torch.arange(length, device=device)

    # embed the sequence
    amino_acids = [amino_acids_by_letter[a] for a in sequence]
    sequence_onehot = torch.stack([aa.one_hot_code for aa in amino_acids]).to(device=device)
    aatype = torch.tensor([aa.index for aa in amino_acids], device=device)
    blosum62 = _get_blosum_encoding(aatype, 62, device)

    # torsion angles (used or not in AA)
    torsion_angles_mask = torch.ones((length, 7), device=device)
    torsion_angles_mask[:, 3:] = torch.tensor([chi_angles_mask[i] for i in aatype], device=device)

    # atoms mask
    atom14_gt_exists = torch.tensor(numpy.array([restype_atom14_mask[i] for i in aatype]), device=device)

    return make_atom14_masks({
        "aatype": aatype,
        "sequence_onehot": sequence_onehot,
        "blosum62": blosum62,
        "residue_numbers": residue_numbers,
        "torsion_angles_mask": torsion_angles_mask,
        "atom14_gt_exists": atom14_gt_exists,
    })


def _read_residue_data(residues: List[Residue]) -> Dict[str, torch.Tensor]:
    """
    Convert residues from a structure into a format that SwiftMHC can work with.
    (these are mostly openfold formats, created by openfold code)

    Args:
        residues: from the structure

    Returns:
        residue_numbers: [len] numbers of the residue as in the structure
        aatype: [len] sequence, indices of amino acids
        sequence_onehot: [len, 22] sequence, one-hot encoded amino acids
        blosum62: [len, 20] sequence, BLOSUM62 encoded amino acids
        backbone_rigid_tensor: [len, 4, 4] 4x4 representation of the backbone frames
        torsion_angles_sin_cos: [len, 7, 2] representations of the torsion angles (one sin & cos per angle)
        alt_torsion_angles_sin_cos: [len, 7, 2] representations of the alternative torsion angles (one sin & cos per angle)
        torsion_angles_mask: [len, 7] which torsion angles each residue has (openfold format)
        atom14_gt_exists: [len, 14] which atoms each residue has (openfold 14 format)
        atom14_gt_positions: [len, 14, 3] atom positions (openfold 14 format)
        atom14_alt_gt_positions: [len, 14, 3] alternative atom positions (openfold 14 format)
        residx_atom14_to_atom37: [len, 14] per residue, conversion table from openfold 14 to openfold 37 atom format
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # embed the sequence
    amino_acids = [amino_acids_by_code[r.get_resname()] for r in residues]
    sequence_onehot = torch.stack([aa.one_hot_code for aa in amino_acids]).to(device=device)
    aatype = torch.tensor([aa.index for aa in amino_acids], device=device)
    blosum62 = _get_blosum_encoding(aatype, 62, device)

    # get atom positions and mask
    atom14_positions = []
    atom14_mask = []
    residue_numbers = []
    for residue_index, residue in enumerate(residues):
        p, m = get_atom14_positions(residue)
        atom14_positions.append(p.float())
        atom14_mask.append(m)
        residue_numbers.append(residue.get_id()[1])

    atom14_positions = torch.stack(atom14_positions).to(device=device)
    atom14_mask = torch.stack(atom14_mask).to(device=device)
    residue_numbers = torch.tensor(residue_numbers, device=device)

    # convert to atom 37 format, for the frames and torsion angles
    protein = {
        "residue_numbers": residue_numbers,
        "aatype": aatype,
        "sequence_onehot": sequence_onehot,
        "blosum62": blosum62,
    }

    protein = make_atom14_masks(protein)

    atom37_positions = atom14_to_atom37(atom14_positions, protein)
    atom37_mask = atom14_to_atom37(atom14_mask.unsqueeze(-1), protein)[..., 0]

    protein["atom14_atom_exists"] = atom14_mask
    protein["atom37_atom_exists"] = atom37_mask

    protein["all_atom_mask"] = atom37_mask
    protein["all_atom_positions"] = atom37_positions

    # get frames, torsion angles and alternative positions
    protein = atom37_to_frames(protein)
    protein = atom37_to_torsion_angles("")(protein)
    protein = get_backbone_frames(protein)
    protein = make_atom14_positions(protein)

    return protein


def _create_proximities(residues1: List[Residue], residues2: List[Residue]) -> torch.Tensor:
    """
    Create a proximity matrix from two lists of residues from a structure.
    proximity = 1.0 / (1.0 + shortest_interatomic_distance)

    Args:
        residues1: residues to be placed on dimension 0 of the matrix
        residues2: residues to be placed on dimension 1 of the matrix
    Returns:
        [len1, len2, 1] proximity matrix
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # allocate memory
    residue_distances = torch.empty((len(residues1), len(residues2), 1), dtype=torch.float32, device=device)

    # get atomic coordinates
    atom_positions1 = [torch.tensor(numpy.array([atom.coord for atom in residue.get_atoms()]), device=device)
                       for residue in residues1]
    atom_positions2 = [torch.tensor(numpy.array([atom.coord for atom in residue.get_atoms()]), device=device)
                       for residue in residues2]

    # calculate distance matrix, using the shortest interatomic distance between two residues
    for i in range(len(residues1)):
        for j in range(len(residues2)):

            atomic_distances_ij = torch.cdist(atom_positions1[i], atom_positions2[j], p=2)

            min_distance = torch.min(atomic_distances_ij).item()

            residue_distances[i, j, 0] = min_distance

    # convert to proximity matrix
    return 1.0 / (1.0 + residue_distances)


def _pymol_superpose(mobile_path: str, target_path: str) -> Tuple[str, str]:
    """
    Superpose a structure onto another structure in PYMOL and create an alignment.

    Args:
        mobile_path: PDB structure, to be superposed
        target_path: PDB structure, to be superposed on
    Returns:
        a path to the superposed PDB structure
        and a path to the clustal alignment (.aln) file
    """

    # define output paths
    name = os.path.basename(mobile_path)
    pdb_output_path = f"superposed-{name}"
    alignment_output_path = f"{pdb_output_path}.aln"

    # init PYMOL
    pymol_cmd.reinitialize()

    # load structures
    pymol_cmd.load(mobile_path, 'mobile')
    pymol_cmd.load(target_path, 'target')

    # superpose
    r = pymol_cmd.align("mobile", "target", object="alignment")
    if r[1] == 0:
        raise ValueError("No residues aligned")

    # save output
    pymol_cmd.save(pdb_output_path, selection="mobile", format="pdb")
    pymol_cmd.save(alignment_output_path, selection="alignment", format="aln")

    # clean up
    pymol_cmd.remove("all")

    return pdb_output_path, alignment_output_path


def _find_model_as_bytes(
    models_path: str,
    model_id: str,
) -> bytes:
    """
    Handles the various ways in which models are stored in directories, subdirectories and tar files.
    This function searches under the given path for a model identified by the given id.

    Args:
        models_path: directory or tarball to search under
        model_id: identifier for the model to search
    Returns:
        the byte contents of the PDB file
    """

    # expect the PDB extension
    model_name = f"{model_id}.pdb"

    # expect at least this many bytes in a PDB file
    min_bytes = 10

    # search under directory
    if os.path.isdir(models_path):

        # search direct children
        model_path = os.path.join(models_path, model_name)
        if os.path.isfile(model_path):
            with open(model_path, 'rb') as f:
                bs = f.read()
                if len(bs) < min_bytes:
                    raise ValueError(f"{len(bs)} bytes in {model_path}")
                return bs

        # search in subdirs named after the BA-identifier
        elif model_id.startswith("BA-"):
            number = int(model_id[3:])

            subset_start = 1000 * floor(number / 1000) + 1
            subset_end = 1000 * ceil(number / 1000)

            subdir_name = f"{subset_start}_{subset_end}"
            model_path = os.path.join(models_path, subdir_name, model_id, "pdb", f"{model_id}.pdb")
            if os.path.isfile(model_path):
                with open(model_path, 'rb') as f:
                    bs = f.read()
                    if len(bs) < min_bytes:
                        raise ValueError(f"{len(bs)} bytes in {model_path}")
                    return bs

    # search in tarball (slow)
    elif models_path.endswith("tar"):
        with tarfile.open(models_path, 'r') as tf:
            for filename in tf.getnames():
                if filename.endswith(model_name):
                    with tf.extractfile(filename) as f:
                        bs = f.read()
                        if len(bs) < min_bytes:
                            raise ValueError(f"{len(bs)} bytes in {model_path}")
                        return bs

    # if really nothing is found
    raise FileNotFoundError(f"Cannot find {model_id} under {models_path}")


def _get_masked_structure(
    model_bytes: bytes,
    reference_structure_path: str,
    reference_masks: Dict[str, List[ResidueMaskType]],
) -> Tuple[Structure, Dict[str, List[Tuple[Residue, bool]]]]:
    """
    Mask a structure, according to the given mask.

    Args:
        model_bytes: contents of the model PDB
        reference_structure_path: structure, to which the mask applies, the model will be aligned to this
        reference_masks: masks that apply to the reference structure, these will be used to mask the given model
    Returns:
        the biopython structure, resulting from the model bytes
        and a dictionary, that contains a list of masked residues per structure
    """

    # need a pdb parser
    pdb_parser = PDBParser()

    # write model to disk
    model_path = f"{uuid4().hex}.pdb"
    with open(model_path, 'wb') as f:
        f.write(model_bytes)

    if len(list(pdb_parser.get_structure("model",model_path).get_residues())) == 0:
        os.remove(model_path)
        raise ValueError(f"no residues in {model_path}")

    # superpose with pymol
    try:
        superposed_model_path, alignment_path = _pymol_superpose(model_path, reference_structure_path)
    finally:
        os.remove(model_path)

    # parse structures and map, according to the pymol alignment
    try:
        superposed_structure = pdb_parser.get_structure("mobile", superposed_model_path)
        reference_structure = pdb_parser.get_structure("target", reference_structure_path)

        if len(list(superposed_structure.get_residues())) == 0:
            raise ValueError(f"no residues in {superposed_model_path}")

        if len(list(reference_structure.get_residues())) == 0:
            raise ValueError(f"no residues in {reference_structure_path}")

        alignment = _map_alignment(alignment_path, (superposed_structure, reference_structure))
    finally:
        os.remove(superposed_model_path)
        os.remove(alignment_path)

    # use the reference structure to map the masks to the model
    mask_result = {}
    for mask_name, reference_mask in reference_masks.items():

        # first, set all to False
        masked_residues = [[residue, False] for residue in superposed_structure.get_residues()]

        # residues, that match with the reference mask, will be set to True
        for chain_id, residue_number, amino_acid in reference_mask:

            # locate the masked residue in the reference structure
            matching_residues = [residue for residue in reference_structure.get_residues()
                                 if residue.get_parent().get_id() == chain_id and
                                    residue.get_id() == (' ', residue_number, ' ')]
            if len(matching_residues) == 0:
                raise ValueError(f"The mask has residue {chain_id},{residue_number}, but the reference structure doesn't")

            reference_residue = matching_residues[0]

            if reference_residue.get_resname() != amino_acid.three_letter_code.upper():
                raise ValueError(
                    f"reference structure contains amino acid {reference_residue.get_resname()} at chain {chain_id} position {residue_number},"
                    f"but the mask has {amino_acid.three_letter_code} there."
                )

            # locate the reference residue in the alignment
            superposed_residue = [rsup for rsup, rref in alignment if rref == reference_residue][0]
            if superposed_residue is not None:

                # set True on the model residue, that was aligned to the reference residue
                masked_residue_index = [i for i in range(len(masked_residues))
                                        if masked_residues[i][0] == superposed_residue][0]

                _log.debug(f"true masking {masked_residue_index}th residue {superposed_residue.get_full_id()} {superposed_residue.get_resname()} in superposed as {chain_id} {residue_number} {amino_acid.three_letter_code}")

                masked_residues[masked_residue_index][1] = True

        mask_result[mask_name] = masked_residues

    return superposed_structure, mask_result


def _k_to_affinity(k: float) -> float:
    """
    The formula used to comvert Kd / IC50 to affinity.
    """

    if k == 0.0:
        raise ValueError(f"k is zero")

    return 1.0 - log(k) / log(50000)


# < 500 nM means BINDING, otherwise not
affinity_binding_threshold = _k_to_affinity(500.0)


def _interpret_target(target: Union[str, float]) -> Tuple[Union[float, None], bool, bool, Union[ComplexClass, None]]:
    """
    target can be anything, decide that here.

    Args:
        target: the target value in the data

    Returns:
        affinity
        does affinity have a less-than inequality y/n
        does affinity have a greater-than inequality y/n
        class BINDING/NONBINDING
    """

    # init to default
    affinity = None
    affinity_lt = False
    affinity_gt = False
    class_ = None

    if isinstance(target, float):
        affinity = _k_to_affinity(target)

    elif target[0].isdigit():
        affinity = _k_to_affinity(float(target))

    elif target.startswith("<"):
        affinity = _k_to_affinity(float(target[1:]))
        affinity_gt = True

    elif target.startswith(">"):
        affinity = _k_to_affinity(float(target[1:]))
        affinity_lt = True

    else:
        class_ = ComplexClass.from_string(target)

    # we can derive the class from the affinity
    # < 500 nM means BINDING, otherwise not
    if affinity is not None:
        if affinity > affinity_binding_threshold and not affinity_lt:
            class_ = ComplexClass.BINDING

        elif affinity <= affinity_binding_threshold and not affinity_gt:
            class_ = ComplexClass.NONBINDING

    return affinity, affinity_lt, affinity_gt, class_


def _generate_structure_data(
    model_bytes: bytes,
    reference_structure_path: str,
    protein_self_mask_path: str,
    protein_cross_mask_path: str,
    allele_name: str,

) -> Tuple[Dict[str, torch.Tensor], Union[Dict[str, torch.Tensor], None]]:
    """
    Get all the data from the structure and put it in a hdf5 storable format.

    Args:
        model_bytes: the pdb model as bytes sequence
        reference_structure_path: a reference structure, its sequence must match with the masks
        protein_self_mask_path: a text file that lists the residues that must be set to true in the mask
        protein_cross_mask_path: a text file that lists the residues that must be set to true in the mask
        allele_name: name of protein allele
    Returns:
        the protein:
            residue_numbers: [len] numbers of the residue as in the structure
            aatype: [len] sequence, indices of amino acids
            sequence_onehot: [len, 22] sequence, one-hot encoded amino acids
            blosum62: [len, 20] sequence, BLOSUM62 encoded amino acids
            backbone_rigid_tensor: [len, 4, 4] 4x4 representation of the backbone frames
            torsion_angles_sin_cos: [len, 7, 2]
            alt_torsion_angles_sin_cos: [len, 7, 2]
            torsion_angles_mask: [len, 7]
            atom14_gt_exists: [len, 14]
            atom14_gt_positions: [len, 14, 3]
            atom14_alt_gt_positions: [len, 14, 3]
            residx_atom14_to_atom37: [len, 14]
            proximities: [len, len, 1]
            allele_name: byte sequence 

        the peptide: (optional)
            residue_numbers: [len] numbers of the residue as in the structure
            aatype: [len] sequence, indices of amino acids
            sequence_onehot: [len, 22] sequence, one-hot encoded amino acids
            blosum62: [len, 20] sequence, BLOSUM62 encoded amino acids
            backbone_rigid_tensor: [len, 4, 4] 4x4 representation of the backbone frames
            torsion_angles_sin_cos: [len, 7, 2]
            alt_torsion_angles_sin_cos: [len, 7, 2]
            torsion_angles_mask: [len, 7]
            atom14_gt_exists: [len, 14]
            atom14_gt_positions: [len, 14, 3]
            atom14_alt_gt_positions: [len, 14, 3]
            residx_atom14_to_atom37: [len, 14]

    """

    # parse the mask files
    protein_residues_self_mask = _read_mask_data(protein_self_mask_path)
    protein_residues_cross_mask = _read_mask_data(protein_cross_mask_path)

    # apply the masks to the MHC model
    structure, masked_residues_dict = _get_masked_structure(
        model_bytes,
        reference_structure_path,
        {"self": protein_residues_self_mask, "cross": protein_residues_cross_mask},
    )
    self_masked_protein_residues = [(r, m) for r, m in masked_residues_dict["self"] if r.get_parent().get_id() == "M"]
    cross_masked_protein_residues = [(r, m) for r, m in masked_residues_dict["cross"] if r.get_parent().get_id() == "M"]

    # locate protein (chain M)
    chains_by_id = {c.id: c for c in structure.get_chains()}
    if "M" not in chains_by_id:
        raise ValueError(f"missing protein chain M in {id_}, present are {chains_by_id.keys()}")

    # order by residue number
    protein_residues = [r for r, m in self_masked_protein_residues]

    # remove the residues that are completely outside of mask range
    combo_mask = numpy.logical_or([m for r, m in self_masked_protein_residues ],
                                  [m for r, m in cross_masked_protein_residues])
    combo_mask_nonzero = combo_mask.nonzero()[0]

    mask_start = combo_mask_nonzero.min()
    mask_end = combo_mask_nonzero.max() + 1

    # apply the limiting protein range, reducing the size of the data that needs to be generated.
    self_residues_mask = [m for r, m in self_masked_protein_residues[mask_start: mask_end]]
    cross_residues_mask = [m for r, m in cross_masked_protein_residues[mask_start: mask_end]]
    protein_residues = protein_residues[mask_start: mask_end]
    if len(protein_residues) < 80:
        raise ValueError(f"got only {len(protein_residues)} protein residues")

    # derive data from protein residues
    protein_data = _read_residue_data(protein_residues)
    protein_data["cross_residues_mask"] = cross_residues_mask
    protein_data["self_residues_mask"] = self_residues_mask

    # proximities within protein
    protein_proximities = _create_proximities(protein_residues, protein_residues)
    protein_data["proximities"] = protein_proximities

    # allele
    protein_data["allele_name"] = numpy.array(allele_name.encode("utf_8"))

    # peptide is optional
    peptide_data = None
    if "P" in chains_by_id:

        # get residues from the peptide chain P
        peptide_chain = chains_by_id["P"]
        peptide_residues = list(peptide_chain.get_residues())
        if len(peptide_residues) < 3:
            raise ValueError(f"got only {len(peptide_residues)} peptide residues")

        peptide_data = _read_residue_data(peptide_residues)

    return protein_data, peptide_data


def preprocess(
    table_path: str,
    models_path: str,
    protein_self_mask_path: str,
    protein_cross_mask_path: str,
    output_path: str,
    reference_structure_path: str,
):
    """
    Preprocess p-MHC-I data, to be used in SwiftMHC.

    Args:
        table_path: CSV input data table, containing columns: ID (of complex),
                    measurement_value (optional, IC50, Kd or BINDING/NONBINDING/POSITIVE/NEGATIVE),
                    allele (optional, name of MHC allele)
        models_path: directory or tarball, to search for models with the IDs from the input table
        protein_self_mask_path: mask file to be used for self attention
        protein_cross_mask_path: mask file to be used for cross attention
        output_path: HDF5 file, to store preprocessed data
        reference_structure_path: structure to align the models to and where the masks apply to
    """

    # in case we're writing to an existing HDF5 file:
    entries_present = set([])
    if os.path.isfile(output_path):
        with h5py.File(output_path, 'r') as output_file:
            entries_present = set(output_file.keys())

    _log.debug(f"{len(entries_present)} entries already present in {output_path}")

    # the table with non-structural data:
    # - peptide sequence
    # - affinity / class
    # - allele name
    table = pandas.read_csv(table_path)

    # here we store temporary data, to be removed after preprocessing:
    tmp_hdf5_path = os.path.join(gettempdir(), f"preprocess-tmp-{uuid4()}.hdf5")

    # iterate through the table
    for table_index, row in table.iterrows():

        # retrieve ID from table
        id_ = row["ID"]
        if id_ in entries_present:
            _log.error(f"duplicate ID in table: '{id_}'")

        # read the affinity data from the table
        affinity_lt = False
        affinity_gt = False
        affinity = None
        class_ = None
        try:
            if "measurement_value" in row:
                affinity, affinity_lt, affinity_gt, class_ = _interpret_target(row["measurement_value"])

                # keep in mind that we do 1 - 50000log(IC50),
                # thus the inequality must be flipped
                if "measurement_inequality" in row:
                    if row["measurement_inequality"] == "<":
                        affinity_gt = True

                    elif row["measurement_inequality"] == ">":
                        affinity_lt = True

            elif "affinity" in row:
                affinity = row["affinity"]

            if "class" in row:
                class_ = row["class"]
        except:
            _log.exception(f"on {id_}")

        # this information is mandatory
        allele = row["allele"]
        peptide_sequence = row["peptide"]

        # find the pdb file
        # for binders a target structure is needed, that contains both MHC and peptide
        # for nonbinders, the MHC structure is sufficient for prediction
        _log.debug(f"finding model for {id_}")
        include_peptide_structure = True
        try:
            model_bytes = _find_model_as_bytes(models_path, id_)
        except (KeyError, FileNotFoundError):

            # at this point, assume that the model is not available
            if class_ == ComplexClass.BINDING:

                _log.exception(f"cannot get structure for {id_}")
                continue
            else:
                # peptide structure is not needed, as NONBINDING
                include_peptide_structure = False

        try:
            if include_peptide_structure:

                # peptide structure is needed, thus load the entire model
                protein_data, peptide_data = _generate_structure_data(
                    model_bytes,
                    reference_structure_path,
                    protein_self_mask_path,
                    protein_cross_mask_path,
                    allele,
                )
            else:
                # not including the peptide structure,
                # check whether the protein structure was already preprocessed
                if _has_protein_data(tmp_hdf5_path, allele):

                    protein_data = _load_protein_data(tmp_hdf5_path, allele)
                else:
                    # if not, preprocess the protein once, reuse the data later from the temporary file
                    model_bytes = _find_model_as_bytes(models_path, allele)
                    protein_data, _ = _generate_structure_data(
                        model_bytes,
                        reference_structure_path,
                        protein_self_mask_path,
                        protein_cross_mask_path,
                        allele,
                    )
                    _save_protein_data(tmp_hdf5_path, allele, protein_data)

                # generate the peptide sequence data, even if the structural data is not used
                peptide_data = _make_sequence_data(peptide_sequence)

            # write the data that we found, to the hdf5 file
            _write_preprocessed_data(
                output_path,
                id_,
                protein_data,
                peptide_data,
                affinity,
                affinity_lt,
                affinity_gt,
                class_,
            )
        except:
            # this case will be skipped
            _log.exception(f"on {id_}")
            continue

    # clean up temporary files after the loop is done and everything is preprocessed:
    if os.path.isfile(tmp_hdf5_path):
        os.remove(tmp_hdf5_path)
