from tempfile import gettempdir
import logging

import numpy
import torch
from Bio.PDB.MMCIFParser import MMCIFParser

from openfold.np.residue_constants import atom_order as restype_atom37_order, restype_atom37_mask
from openfold.config import model_config
from openfold.data.data_modules import OpenFoldSingleDataset
from openfold.data.feature_pipeline import FeaturePipeline
from openfold.data.data_pipeline import DataPipeline
from openfold.data import mmcif_parsing
from openfold.data import data_transforms
from openfold.utils.rigid_utils import Rigid

from mock import patch


_log = logging.getLogger(__name__)


@patch("openfold.data.input_pipeline.nonensembled_transform_fns")
def test_mmcif_atom_coords_unit_angstrom(mock_transform_fns):

    mock_transform_fns.return_value = [
        data_transforms.cast_to_64bit_ints,
        data_transforms.squeeze_features,
        data_transforms.make_seq_mask,
        data_transforms.make_atom14_masks,
        data_transforms.make_atom14_positions,
        data_transforms.atom37_to_frames,
        data_transforms.atom37_to_torsion_angles(""),
        data_transforms.make_pseudo_beta(""),
        data_transforms.get_backbone_frames,
        data_transforms.get_chi_angles,
    ]

    feature_names = [
        "msa",
        "aatype",
        "all_atom_mask",
        "all_atom_positions",
        "deletion_matrix",
        "between_segment_residues",
    ]

    config = model_config(
        "initial_training",
        train=True,
    )
    config["train"] = {
        "crop_size": None,
        "supervised": None,
        "max_msa_clusters": 0,
        "max_extra_msa": 0,
        "fixed_size": None,
        "max_templates": 0,
    }
    config["common"] = {
        "use_templates": False,
        "max_recycling_iters": 0,
        "reduce_msa_clusters_by_max_templates": False,
        "resample_msa_in_recycling": False,
        "msa_cluster_features": False,
        "feat": [],
        "unsupervised_features": feature_names,
    }
    config["supervised"] = {
        "clamp_prob": False,
    }

    data_pipeline = DataPipeline(None)

    feature_pipeline = FeaturePipeline(config)

    with open("tests/data/101M.cif", 'rt') as mmcif_file:
        mmcif_object = mmcif_parsing.parse(file_id="101M", mmcif_string=mmcif_file.read())

    data = data_pipeline.process_mmcif(
        mmcif=mmcif_object.mmcif_object,
        alignment_dir=gettempdir(),
        chain_id="A",
        alignment_index=None,
    )

    features = feature_pipeline.process_features(data, "train")

    mmcif_parser = MMCIFParser()

    structure = mmcif_parser.get_structure("101M", "tests/data/101M.cif")

    chain = list(structure.get_chains())[0]

    all_atom_positions = features["all_atom_positions"]

    bb_frames = Rigid.from_tensor_4x4(features["backbone_rigid_tensor"][..., 0])
    bb_exists = features["backbone_rigid_mask"][..., 0]

    compared_count = 0
    for residue_index, residue in enumerate(chain.get_residues()):
        if residue_index < all_atom_positions.shape[0]:

            if bb_exists[residue_index]:
                for atom in residue.get_atoms():
                    if atom.name == "CA":
                        assert tuple(atom.coord) == tuple(bb_frames.get_trans()[residue_index])
                        break
                else:
                    raise ValueError(f"residue {residue_index} has no CA")

            for atom in residue.get_atoms():
                if atom.name in restype_atom37_order:
                    atom_index = restype_atom37_order[atom.name]

                    assert tuple(all_atom_positions[residue_index][atom_index]) == tuple(atom.coord)
                    compared_count += 1

    assert compared_count > 1000
