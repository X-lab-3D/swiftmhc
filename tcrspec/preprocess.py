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
from Bio.PDB.Polypeptide import is_aa, one_to_three
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
PREPROCESS_PEPTIDE_NAME = "peptide"


def _write_preprocessed_data(hdf5_path: str, storage_id: str,
                             protein_data: Dict[str, torch.Tensor],
                             peptide_data: Optional[Dict[str, torch.Tensor]] = None,
                             target: Optional[Union[float, ComplexClass]] = None):
    """
    Output preprocessed protein-peptide data to and hdf5 file.

    Args:
        hdf5_path: path to output file
        storage_id: id to store the entry under as an hdf5 group
        protein_data: result output by '_read_residue_data' function, on protein residues
        peptide_data: result output by '_read_residue_data' function, on peptide residues
        target: a number(Kd) or a class(BINDING/NONBINDING) to express binding affinity
    """

    with h5py.File(hdf5_path, 'a') as hdf5_file:

        storage_group = hdf5_file.require_group(storage_id)

        # store target data
        if isinstance(target, float):
            storage_group.create_dataset(PREPROCESS_KD_NAME, data=target)

        elif isinstance(target, str):
            cls = ComplexClass.from_string(target)
            storage_group.create_dataset(PREPROCESS_CLASS_NAME, data=int(cls))

        elif isinstance(target, ComplexClass):
            storage_group.create_dataset(PREPROCESS_CLASS_NAME, data=int(target))

        elif target is not None:
            raise TypeError(type(target))

        # store protein data
        protein_group = storage_group.require_group(PREPROCESS_PROTEIN_NAME)
        for field_name, field_data in protein_data.items():
            if isinstance(field_data, torch.Tensor):
                protein_group.create_dataset(field_name, data=field_data.cpu(), compression="lzf")
            else:
                protein_group.create_dataset(field_name, data=field_data)

        if peptide_data is not None:
            peptide_group = storage_group.require_group(PREPROCESS_PEPTIDE_NAME)
            for field_name, field_data in peptide_data.items():
                if isinstance(field_data, torch.Tensor):
                    peptide_group.create_dataset(field_name, data=field_data.cpu(), compression="lzf")
                else:
                    peptide_group.create_dataset(field_name, data=field_data)


ResidueMaskType = Tuple[str, int, AminoAcid]

def _read_mask_data(path: str) -> List[ResidueMaskType]:
    """
    Read from the mask TSV file, which residues in the PDB file should be marked as True.

    Format: CHAIN_ID  RESIDUE_NUMBER  AMINO_ACID_THREE_LETTER_CODE

    Lines starting in '#' will be ignored

    Args:
        path: input TSV file
    Returns:
        list of residues to be selected
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


def _get_blosum_encoding(amino_acid_indexes: List[int], blosum_index: int) -> torch.Tensor:
    """
    Convert amino acids to BLOSUM encoding

    Arguments:
        amino_acid_indexes: order of numbers 0 to 19, coding for the amino acids
        blosum_index: identifies the type of BLOSUM matrix to use
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

    # search in tarball
    elif models_path.endswith("tar.xz"):
        model_path = os.path.join(models_path, model_name)
        with tarfile.open(models_path, 'r:xz') as tf:
            with tf.extractfile(model_path) as f:
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
        a dictionary, that contains a list of masked residues per structure
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
            reference_residue = [residue for residue in reference_structure.get_residues()
                                 if residue.get_parent().get_id() == chain_id and
                                    residue.get_id() == (' ', residue_number, ' ')][0]

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
        table_path: CSV input data table, containing columns: ID (of complex), measurement_value (optional, Kd or BINDING/NONBINDING), allele (optional, name of MHC allele)
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

    # parse the mask files
    protein_residues_self_mask = _read_mask_data(protein_self_mask_path)
    protein_residues_cross_mask = _read_mask_data(protein_cross_mask_path)

    table = pandas.read_csv(table_path)

    for table_index, row in table.iterrows():

        id_ = row["ID"]
        if id_ in entries_present:
            continue

        _log.debug(f"preprocessing {id_}")

        if "measurement_value" in row:
            target = row["measurement_value"]
        else:
            target = None

        if "allele" in row:
            allele = row["allele"]
        else:
            allele = None

        # find the pdb file
        try:
            model_bytes = _find_model_as_bytes(models_path, id_)
        except (KeyError, FileNotFoundError):
            _log.exception(f"cannot get structure for {id_}")
            continue

        try:
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

            # proximities within protein
            protein_proximities = _create_proximities(protein_residues, protein_residues)
            protein_data["proximities"] = protein_proximities

            # SwiftMHC doesn't need the allele name to function.
            # We store it for administrative purposes.
            if allele is not None:
                protein_data["allele_name"] = numpy.array(allele.encode("utf_8"))

            # get residues from the peptide (chain P, optional)
            if "P" in chains_by_id:
                peptide_chain = chains_by_id["P"]
                peptide_residues = list(peptide_chain.get_residues())
                if len(peptide_residues) < 3:
                    raise ValueError(f"{id_}: got only {len(peptide_residues)} peptide residues")

                peptide_data = _read_residue_data(peptide_residues)
            else:
                peptide_data = None

            # write the data that we found
            _write_preprocessed_data(output_path, id_,
                                     protein_data,
                                     peptide_data,
                                     target)
        except:
            _log.exception(f"on {id_}")
            continue
