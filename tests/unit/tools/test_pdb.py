from pdb2sql import pdb2sql

import torch

from tcrspec.tools.pdb import get_residues, get_residue_transformations, get_residue_proximities


def test_residue_properties():

    pdb = pdb2sql("tests/data/1crn.pdb")
    try:
        residues = get_residues(pdb)
    finally:
        pdb._close()

    proximities = get_residue_proximities(residues)

    assert proximities.shape == (46, 46), f"output shape is {proximities.shape}"
    assert torch.all(proximities >= 0.0)
    assert torch.all(proximities <= 1.0)

    translations, rotations = get_residue_transformations(residues)
    assert translations.shape == (46, 3), f"output shape is {proximities.shape}"
    assert rotations.shape == (46, 3, 3), f"output shape is {proximities.shape}"
