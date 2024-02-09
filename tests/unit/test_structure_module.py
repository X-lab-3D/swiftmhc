import torch
from Bio.PDB.PDBParser import PDBParser

from swiftmhc.tools.pdb import get_atom14_positions
from swiftmhc.modules.cross_structure_module import CrossStructureModule


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

    mask = torch.ones(xyz.shape[:-2])

    omegas = m.calculate_omegas_from_positions(xyz, mask)

    eps = 0.05

    for index in range(omegas.shape[0]):

        if residues[index + 1].get_resname() == "PRO":
            continue

        sin, cos = omegas[index]

        # omega must be close to pi radials, 180 degrees
        assert abs(cos + 1.0) < eps, (residues[index], residues[index + 1], cos.item())
        assert abs(sin) < eps, (residues[index], residue[index + 1], sin.item())
