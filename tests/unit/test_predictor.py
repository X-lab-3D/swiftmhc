import logging
import random
from typing import List
import torch
from openfold.np import residue_constants as rc
from swiftmhc.config import config
from swiftmhc.loss import get_loss
from swiftmhc.modules.predictor import Predictor


_log = logging.getLogger(__name__)


def random_mask(maxlen: int) -> List[bool]:
    length = random.randrange(maxlen)
    diff = maxlen - length

    return [True] * length + [False] * diff


def test_predictor():
    n_complexes = 10
    peptide_maxlen = 16
    protein_maxlen = 200

    data = {
        "peptide_sequence_onehot": torch.rand(n_complexes, peptide_maxlen, 32),
        "peptide_self_residues_mask": torch.tensor(
            [random_mask(peptide_maxlen) for __ in range(n_complexes)]
        ),
        "peptide_cross_residues_mask": torch.tensor(
            [random_mask(peptide_maxlen) for __ in range(n_complexes)]
        ),
        "peptide_aatype": torch.tensor(
            [[random.randrange(20) for _ in range(peptide_maxlen)] for __ in range(n_complexes)]
        ),
        "peptide_torsion_angles_mask": torch.tensor(
            [
                [[bool(random.randrange(2)) for ___ in range(7)] for _ in range(peptide_maxlen)]
                for __ in range(n_complexes)
            ]
        ),
        "peptide_torsion_angles_sin_cos": torch.rand(n_complexes, peptide_maxlen, 7, 2),
        "peptide_alt_torsion_angles_sin_cos": torch.rand(n_complexes, peptide_maxlen, 7, 2),
        "peptide_backbone_rigid_tensor": torch.rand(n_complexes, peptide_maxlen, 4, 4),
        "peptide_atom14_gt_positions": torch.rand(n_complexes, peptide_maxlen, 14, 3),
        "peptide_atom14_alt_gt_positions": torch.rand(n_complexes, peptide_maxlen, 14, 3),
        "peptide_atom14_gt_exists": torch.tensor(
            [
                [[bool(random.randrange(2)) for ___ in range(14)] for _ in range(peptide_maxlen)]
                for __ in range(n_complexes)
            ]
        ),
        "peptide_all_atom_positions": torch.rand(n_complexes, peptide_maxlen, 37, 3),
        "peptide_all_atom_mask": torch.tensor(
            [
                [[bool(random.randrange(2)) for ___ in range(37)] for _ in range(peptide_maxlen)]
                for __ in range(n_complexes)
            ]
        ),
        "peptide_residue_index": torch.tensor([range(peptide_maxlen) for __ in range(n_complexes)]),
        "protein_sequence_onehot": torch.rand(n_complexes, protein_maxlen, 32),
        "protein_cross_residues_mask": torch.tensor(
            [random_mask(protein_maxlen) for __ in range(n_complexes)]
        ),
        "protein_self_residues_mask": torch.tensor(
            [random_mask(protein_maxlen) for __ in range(n_complexes)]
        ),
        "protein_backbone_rigid_tensor": torch.rand(n_complexes, protein_maxlen, 4, 4),
        "protein_proximities": torch.rand(n_complexes, protein_maxlen, protein_maxlen, 1),
        "protein_aatype": torch.tensor(
            [[random.randrange(20) for _ in range(protein_maxlen)] for __ in range(n_complexes)]
        ),
        "protein_atom14_gt_positions": torch.rand(n_complexes, protein_maxlen, 14, 3),
        "protein_atom14_gt_exists": torch.tensor(
            [
                [[bool(random.randrange(2)) for ___ in range(14)] for _ in range(protein_maxlen)]
                for __ in range(n_complexes)
            ]
        ),
        "protein_all_atom_positions": torch.rand(n_complexes, protein_maxlen, 37, 3),
        "protein_all_atom_mask": torch.tensor(
            [
                [[bool(random.randrange(2)) for ___ in range(37)] for _ in range(protein_maxlen)]
                for __ in range(n_complexes)
            ]
        ),
        "protein_residue_index": torch.tensor([range(protein_maxlen) for __ in range(n_complexes)]),
        "affinity": torch.rand(n_complexes),
        "affinity_lt": torch.zeros(n_complexes, dtype=torch.bool),
        "affinity_gt": torch.zeros(n_complexes, dtype=torch.bool),
    }
    restype_atom14_to_atom37 = []
    for rt in rc.restypes:
        atom_names = rc.restype_name_to_atom14_names[rc.restype_1to3[rt]]
        restype_atom14_to_atom37.append(
            [(rc.atom_order[name] if name else 0) for name in atom_names]
        )

    data["peptide_residx_atom14_to_atom37"] = torch.tensor(restype_atom14_to_atom37)[
        data["peptide_aatype"]
    ]
    data["protein_residx_atom14_to_atom37"] = torch.tensor(restype_atom14_to_atom37)[
        data["protein_aatype"]
    ]

    model = Predictor(config)

    output = model(data)

    losses = get_loss(config.model_type, output, data, True, True, True, True)

    assert "total" in losses
