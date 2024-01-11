from typing import List, Tuple, Union, Optional, Dict
import os
import logging
from math import isinf, floor, ceil
import tarfile
from uuid import uuid4

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
from Bio.PDB.Polypeptide import three_to_one, is_aa, one_to_three
from blosum import BLOSUM

from pymol import cmd as pymol_cmd

from openfold.np.residue_constants import restype_atom37_mask
from openfold.data.data_transforms import (atom37_to_frames,
                                           atom37_to_torsion_angles,
                                           get_backbone_frames,
                                           make_atom14_masks,
                                           make_atom14_positions)
from openfold.utils.feats import atom14_to_atom37

from .tools.pdb import (get_residue_transformations,
                        get_atom14_positions,
                        generate_symmetry_alternative,
                        get_residue_proximities)
from .domain.amino_acid import amino_acids_by_code, canonical_amino_acids
from .models.amino_acid import AminoAcid
from .models.complex import ComplexClass


_log = logging.getLogger(__name__)


PREPROCESS_KD_NAME = "kd"
PREPROCESS_CLASS_NAME = "class"
PREPROCESS_PROTEIN_NAME = "protein"
PREPROCESS_LOOP_NAME = "loop"


def _write_preprocessed_data(hdf5_path: str, storage_id: str,
                             protein_data: Dict[str, torch.Tensor],
                             loop_data: Dict[str, torch.Tensor],
                             target: Optional[Union[float, ComplexClass]] = None):

    with h5py.File(hdf5_path, 'a') as hdf5_file:

        storage_group = hdf5_file.require_group(storage_id)

        if isinstance(target, float):
            storage_group.create_dataset(PREPROCESS_KD_NAME, data=target)

        elif isinstance(target, str):
            cls = ComplexClass.from_string(target)
            storage_group.create_dataset(PREPROCESS_CLASS_NAME, data=int(cls))

        elif isinstance(target, ComplexClass):
            storage_group.create_dataset(PREPROCESS_CLASS_NAME, data=int(target))
        else:
            raise TypeError(type(target))

        protein_group = storage_group.require_group(PREPROCESS_PROTEIN_NAME)
        for field_name, field_data in protein_data.items():
            if isinstance(field_data, torch.Tensor):
                protein_group.create_dataset(field_name, data=field_data.cpu(), compression="lzf")
            else:
                protein_group.create_dataset(field_name, data=field_data)

        loop_group = storage_group.require_group(PREPROCESS_LOOP_NAME)
        for field_name, field_data in loop_data.items():
            if isinstance(field_data, torch.Tensor):
                loop_group.create_dataset(field_name, data=field_data.cpu(), compression="lzf")
            else:
                loop_group.create_dataset(field_name, data=field_data)


def _read_targets_by_id(table_path: str) -> List[Tuple[str, Union[float, ComplexClass]]]:
    """
    Args:
        table_path: points to a csv table
    """

    table = pandas.read_csv(table_path)

    data = []
    for index, row in table.iterrows():
        value = row["measurement_value"]
        id_ = row["ID"]

        try:
            value = float(value)
        except ValueError:
            value = ComplexClass.from_string(value)

        data.append((id_, value))

    return data


def _read_mask_data(path: str) -> List[Tuple[str, int, AminoAcid]]:

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


def _get_blosum_encoding(amino_acid_indexes: List[int], blosum_index: int) -> List[int]:
    """
    Arguments:
        amino_acid_indexes: order of numbers 0 to 19, coding for the amino acids
        blosum_index: identifies the type of BLOSUM matrix to use
    Returns:
        the amino acids encoded by their BLOSUM rows
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


ResidueMaskType = Tuple[str, int, AminoAcid]


def _make_alignment_map(sorted_residues: List[Residue], mask_ids: List[ResidueMaskType]) -> Dict[int, int]:
    """
    Returns: a dictionary, mapping mask residue numbers to the sorted residue numbers
    """

    # do not allow gaps
    aligner = PairwiseAligner(target_internal_open_gap_score=-1e22,
                              target_internal_extend_gap_score=-1e22)

    residues_seq = ""
    for residue in sorted_residues:
        residue_amino_acid = amino_acids_by_code[residue.get_resname()]
        residues_seq += residue_amino_acid.one_letter_code

    # get the sequence of the mask, sort by residue number
    mask_seq = ""
    for chain_id, residue_number, amino_acid in sorted(mask_ids, key=lambda id_: id_[1]):
        mask_seq += amino_acid.one_letter_code

    alignments = aligner.align(residues_seq, mask_seq)
    alignment = alignments[0]
    pid = 100.0 * alignment.score / len(mask_ids)
    if pid < 35.0:
        raise ValueError(f"cannot reliably align mask to structure, identity is only {pid} %")

    # we expect no gaps, so there's only one range
    residues_start = alignment.aligned[0][0][0]
    residues_end = alignment.aligned[0][0][1]

    mask_start = alignment.aligned[1][0][0]
    mask_end = alignment.aligned[1][0][1]

    _log.debug(f"alignment made:\nresidues:  {residues_seq[residues_start: residues_end]}\nmask    :  {mask_seq[mask_start: mask_end]}")

    # map the mask to the residues
    map_ = {}
    alignment_len = residues_end - residues_start
    for alignment_index in range(alignment_len):

        residue_index = residues_start + alignment_index
        residue_number = sorted_residues[residue_index].get_full_id()[-1][1]

        mask_index = mask_start + alignment_index
        mask_number = mask_ids[mask_index][1]

        map_[residue_number] = mask_number

    return map_


def _map_structure_alignment(
    aln_path: str,
    aligned_structures: Tuple[Structure, Structure],

) -> List[Tuple[Union[Residue, None], Union[Residue, None]]]:

    alignment = {}
    with open(aln_path) as handle:
        for record in SeqIO.parse(handle, "clustal"):
            alignment[record.id] = str(record.seq)

    maps = []
    for structure in aligned_structures:
        key = structure.get_id()
        if key not in alignment:
            continue

        sequence = alignment[key]
        residues = [r for r in structure.get_residues() if is_aa(r.get_resname())]

        _log.debug(f"mapping to {len(residues)} {key} residues:\n{sequence}")

        map_ = []
        offset = 0
        for i in range(len(sequence)):
            if sequence[i].isalpha():

                letter = sequence[i]

                if offset >= len(residues):
                    raise ValueError(f"{key} alignment has over {offset} residues, but the structure only has {len(residues)}")

                if letter != 'X' and one_to_three(letter) != residues[offset].get_resname():
                    _log.warning(f"encountered {residues[offset].get_resname()} at {offset}, {one_to_three(letter)} expected")

                map_.append(residues[offset])
                offset += 1
            else:
                map_.append(None)
        maps.append(map_)

    results = [(maps[0][i], maps[1][i]) for i in range(len(maps[0]))]

    return results


def _mask_residues(residues: List[Residue],
                   mask_ids: List[ResidueMaskType],
                   alignment_map: Dict[int, int]) -> torch.Tensor:

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    mask_residue_numbers = [mask_id[1] for mask_id in mask_ids]

    mask = []
    for residue in residues:

        residue_number = residue.get_full_id()[-1][1]

        if residue_number in alignment_map:
            mask_number = alignment_map[residue_number]

            mask.append(mask_number in mask_residue_numbers)
        else:
            mask.append(False)

    mask = torch.tensor(mask, dtype=torch.bool, device=device)

    if not torch.any(mask):
        raise ValueError(f"none found of {mask_ids} in {len(residues)} residues")

    return mask


def _read_residue_data(residues: List[Residue]) -> Dict[str, torch.Tensor]:
    """
    Returns:
        aatype: [len] sequence, indices of amino acids
        sequence_onehot: [len, depth] sequence, one-hot encoded amino acids
        backbone_rigid_tensor: [len, 4, 4] 4x4 representation of the backbone frames
        torsion_angles_sin_cos: [len, 7, 2]
        alt_torsion_angles_sin_cos: [len, 7, 2]
        torsion_angles_mask: [len, 7]
        atom14_gt_exists: [len, 14]
        atom14_gt_positions: [len, 14, 3]
        atom14_alt_gt_positions: [len, 14, 3]
        residx_atom14_to_atom37: [len, 14]
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # embed the sequence
    amino_acids = [amino_acids_by_code[r.get_resname()] for r in residues]
    sequence_onehot = torch.stack([aa.one_hot_code for aa in amino_acids])
    aatype = torch.tensor([aa.index for aa in amino_acids], device=device)

    # get atom positions and mask
    atom14_positions = []
    atom14_mask = []
    residue_numbers = []
    for residue_index, residue in enumerate(residues):
        p, m = get_atom14_positions(residue)
        atom14_positions.append(p.float())
        atom14_mask.append(m)
        residue_numbers.append(residue.get_id()[1])

    atom14_positions = torch.stack(atom14_positions)
    atom14_mask = torch.stack(atom14_mask)
    residue_numbers = torch.tensor(residue_numbers, device=device)

    blosum62 = _get_blosum_encoding(aatype, 62)

    # convert to atom 37 format, for the frames and torsion angles
    protein = {
        "residue_numbers": residue_numbers,
        "aatype": aatype,
        "sequence_onehot": sequence_onehot,
        "blosum62": blosum62
    }
    protein = make_atom14_masks(protein)

    atom37_positions = atom14_to_atom37(atom14_positions, protein)
    protein["all_atom_mask"] = protein["atom37_atom_exists"]
    protein["all_atom_positions"] = atom37_positions

    # get frames, torsion angles and alternative positions
    protein = atom37_to_frames(protein)
    protein = atom37_to_torsion_angles("")(protein)
    protein = get_backbone_frames(protein)
    protein = make_atom14_positions(protein)

    return protein


def _create_symmetry_alternative(chain: Chain) -> Chain:
    alt_chain = Chain(chain.id)

    for residue in chain.get_residues():
        alt_chain.add(generate_symmetry_alternative(residue))

    return alt_chain


def _create_proximities(residues1: List[Residue], residues2: List[Residue]) -> torch.Tensor:

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    residue_distances = torch.empty((len(residues1), len(residues2), 1), dtype=torch.float32)

    atom_positions1 = [torch.tensor(numpy.array([atom.coord for atom in residue.get_atoms()]), device=device)
                       for residue in residues1]
    atom_positions2 = [torch.tensor(numpy.array([atom.coord for atom in residue.get_atoms()]), device=device)
                       for residue in residues2]

    for i in range(len(residues1)):
        for j in range(len(residues2)):

            atomic_distances_ij = torch.cdist(atom_positions1[i], atom_positions2[j], p=2)

            min_distance = torch.min(atomic_distances_ij).item()

            residue_distances[i, j, 0] = min_distance

    return 1.0 / (1.0 + residue_distances)


def _pymol_superpose(mobile_path: str, target_path: str) -> Tuple[str, str]:

    name = os.path.basename(mobile_path)
    pdb_output_path = f"superposed-{name}"
    alignment_output_path = f"{pdb_output_path}.aln"

    pymol_cmd.reinitialize()

    pymol_cmd.load(mobile_path, 'mobile')
    pymol_cmd.load(target_path, 'target')

    r = pymol_cmd.align("mobile", "target", object="alignment")
    if r[1] == 0:
        raise ValueError("No residues aligned")

    pymol_cmd.save(pdb_output_path, selection="mobile", format="pdb")
    pymol_cmd.save(alignment_output_path, selection="alignment", format="aln")

    pymol_cmd.remove("all")

    return pdb_output_path, alignment_output_path


def _find_model_as_bytes(
    models_path: str,
    model_id: str,
) -> bytes:

    model_name = f"{model_id}.pdb"
    if os.path.isdir(models_path):
        model_path = os.path.join(models_path, model_name)
        if os.path.isfile(model_path):
            with open(model_path, 'rb') as f:
                bs = f.read()
                if len(bs) < 10:
                    raise ValueError(f"{len(bs)} bytes in {model_path}")
                return bs

        elif model_id.startswith("BA-"):
            number = int(model_id[3:])

            subset_start = 1000 * floor(number / 1000) + 1
            subset_end = 1000 * ceil(number / 1000)

            subdir_name = f"{subset_start}_{subset_end}"
            model_path = os.path.join(models_path, subdir_name, model_id, "pdb", f"{model_id}.pdb")
            if os.path.isfile(model_path):
                with open(model_path, 'rb') as f:
                    bs = f.read()
                    if len(bs) < 10:
                        raise ValueError(f"{len(bs)} bytes in {model_path}")
                    return bs

    elif models_path.endswith("tar.xz"):
        model_path = os.path.join(models_path, model_name)
        with tarfile.open(models_path, 'r:xz') as tf:
            with tf.extractfile(model_path) as f:
                bs = f.read()
                if len(bs) < 10:
                    raise ValueError(f"{len(bs)} bytes in {model_path}")
                return bs

    raise FileNotFoundError(f"Cannot find {model_id} under {models_path}")


def _get_masked_structure(
    model_bytes: bytes,
    reference_structure_path: str,
    reference_masks: Dict[str, List[ResidueMaskType]],
) -> Tuple[Structure, Dict[str, List[Tuple[Residue, bool]]]]:

    pdb_parser = PDBParser()

    # write model to disk
    model_path = f"{uuid4().hex}.pdb"
    with open(model_path, 'wb') as f:
        f.write(model_bytes)

    if len(list(pdb_parser.get_structure("model",model_path).get_residues())) == 0:
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

        alignment = _map_structure_alignment(alignment_path, (superposed_structure, reference_structure))
    finally:
        os.remove(superposed_model_path)
        os.remove(alignment_path)

    # use the reference structure to map the masks to the model
    mask_result = {}
    for mask_name, reference_mask in reference_masks.items():

        masked_residues = [[residue, False] for residue in superposed_structure.get_residues()]

        for chain_id, residue_number, amino_acid in reference_mask:
            reference_residue = [residue for residue in reference_structure.get_residues()
                                 if residue.get_parent().get_id() == chain_id and
                                    residue.get_id() == (' ', residue_number, ' ')][0]

            if reference_residue.get_resname() != amino_acid.three_letter_code.upper():
                raise ValueError(
                    f"reference structure contains amino acid {reference_residue.get_resname()} at chain {chain_id} position {residue_number},"
                    f"but the mask has {amino_acid.three_letter_code} there."
                )

            superposed_residue = [rsup for rsup, rref in alignment if rref == reference_residue][0]
            if superposed_residue is not None:

                masked_residue_index = [i for i in range(len(masked_residues))
                                        if masked_residues[i][0] == superposed_residue][0]

                _log.debug(f"true masking {masked_residue_index}th residue {superposed_residue.get_full_id()} {superposed_residue.get_resname()} in superposed as {chain_id} {residue_number} {amino_acid.three_letter_code}")

                masked_residues[masked_residue_index][1] = True

        mask_result[mask_name] = masked_residues

    return superposed_structure, mask_result


def preprocess(
    table_path: str,
    models_path: str,
    protein_self_mask_path: str,
    protein_cross_mask_path: str,
    output_path: str,
    reference_structure_path: str,
):
    # in case we're writing to an existing file:
    entries_present = set([])
    if os.path.isfile(output_path):
        with h5py.File(output_path, 'r') as output_file:
            entries_present = set(output_file.keys())

    _log.debug(f"{len(entries_present)} entries already present in {output_path}")

    protein_residues_self_mask = _read_mask_data(protein_self_mask_path)
    protein_residues_cross_mask = _read_mask_data(protein_cross_mask_path)

    table = pandas.read_csv(table_path)

    for table_index, row in table.iterrows():

        id_ = row["ID"]
        if id_ in entries_present:
            continue

        _log.debug(f"preprocessing {id_}")

        target = row["measurement_value"]
        allele = row["allele"]

        # find the pdb file
        try:
            model_bytes = _find_model_as_bytes(models_path, id_)
        except (KeyError, FileNotFoundError):
            _log.exception(f"cannot get structure for {id_}")
            continue

        try:
            structure, masked_residues_dict = _get_masked_structure(
                model_bytes,
                reference_structure_path,
                {"self": protein_residues_self_mask, "cross": protein_residues_cross_mask},
            )
            self_masked_protein_residues = [(r, m) for r, m in masked_residues_dict["self"] if r.get_parent().get_id() == "M"]
            cross_masked_protein_residues = [(r, m) for r, m in masked_residues_dict["cross"] if r.get_parent().get_id() == "M"]

            # locate protein and loop
            chains_by_id = {c.id: c for c in structure.get_chains()}
            if "M" not in chains_by_id:
                raise ValueError(f"missing protein chain M in {id_}, present are {chains_by_id.keys()}")
            if "P" not in chains_by_id:
                raise ValueError(f"missing loop chain P in {id_}, present are {chains_by_id.keys()}")

            # order by residue number
            protein_residues = [r for r, m in self_masked_protein_residues]

            # remove the residues that are completely outside of mask range
            combo_mask = numpy.logical_or([m for r, m in self_masked_protein_residues ],
                                          [m for r, m in cross_masked_protein_residues])
            combo_mask_nonzero = combo_mask.nonzero()[0]
            _log.debug(f"nonzero: {combo_mask_nonzero}")

            mask_start = combo_mask_nonzero.min()
            mask_end = combo_mask_nonzero.max() + 1

            _log.debug(f"{id_}: taking protein residues {mask_start} - {mask_end}")

            # apply the limiting protein range, reducing the size of the data that needs to be generated.
            self_residues_mask = [m for r, m in self_masked_protein_residues[mask_start: mask_end]]
            cross_residues_mask = [m for r, m in cross_masked_protein_residues[mask_start: mask_end]]
            protein_residues = protein_residues[mask_start: mask_end]
            if len(protein_residues) < 80:
                raise ValueError(f"{id_}: got only {len(protein_residues)} protein residues")

            # derive data from protein residues
            protein_data = _read_residue_data(protein_residues)
            protein_data["cross_residues_mask"] = cross_residues_mask
            protein_data["self_residues_mask"] = self_residues_mask

            # get residues from the loop (chain P)
            loop_chain = chains_by_id["P"]
            loop_residues = list(loop_chain.get_residues())
            if len(loop_residues) < 3:
                raise ValueError(f"{id_}: got only {len(loop_residues)} loop residues")

            loop_data = _read_residue_data(loop_residues)

            # proximities within protein
            protein_proximities = _create_proximities(protein_residues, protein_residues)
            protein_data["proximities"] = protein_proximities
            protein_data["allele_name"] = numpy.array(allele.encode("utf_8"))

            _write_preprocessed_data(output_path, id_,
                                     protein_data,
                                     loop_data,
                                     target)
        except:
            _log.exception(f"on {id_}")
            continue
