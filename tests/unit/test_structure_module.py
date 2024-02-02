import torch
from Bio.PDB.PDBParser import PDBParser

from tcrspec.tools.pdb import get_atom14_positions
from tcrspec.modules.cross_structure_module import CrossStructureModule


pdb_parser = PDBParser()


def test_omega_calculation():

    m = CrossStructureModule(32, 16, 32, 2, 4, 8, 0.1, 2, 2, 7, 10, 2, 1e-6)

    pdb = pdb_parser.get_structure("BA-55224", "tests/data/BA-55224.pdb")
    residues = list(pdb[0]["M"].get_residues())

    xyz = []
    for residue in residues:
        pos, mask = get_atom14_positions(residue)
        xyz.append(pos)

    xyz = torch.stack(xyz)

    omegas_unnormalized = m.calculate_omegas_from_positions(xyz)
    omegas = torch.nn.functional.normalize(omegas_unnormalized, dim=-1)

    eps = 0.15

    for index, residue in enumerate(residues):

        if index > 0:

            sin, cos = omegas[index]

            assert abs(abs(cos) - 1.0) < eps, (residues[index - 1], residue, cos.item())
            assert abs(sin) < eps, (residues[index - 1], residue, sin.item())
