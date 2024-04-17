
import torch

from Bio.PDB.PDBParser import PDBParser

from swiftmhc.tools.md import build_modeller
from swiftmhc.preprocess import _read_residue_data


pdb_parser = PDBParser()


def test_modeller():

        pdb = pdb_parser.get_structure("test", "tests/data/1crn.pdb")
        model = list(pdb.get_models())[0]
        chain = list(model.get_chains())[0]
        residues = list(chain.get_residues())

        protein = _read_residue_data(residues)

        length = len(residues)
        max_length = 100
        padding = max_length - length

        residue_numbers = torch.tensor(list(range(length)) + padding * [0])
        aatype = torch.cat((protein['aatype'], torch.zeros(padding, dtype=torch.int)), dim=0)
        atom14_mask = torch.cat((protein['atom14_gt_exists'], torch.zeros(padding, 14, dtype=torch.bool)), dim=0)
        atom14_positions = torch.cat((protein['atom14_gt_positions'], torch.zeros(padding, 14, 3)), dim=0)

        modeller = build_modeller(
            [
                ('A', residue_numbers, aatype, atom14_positions, atom14_mask),
            ]
        )

        modeller.addHydrogens()
