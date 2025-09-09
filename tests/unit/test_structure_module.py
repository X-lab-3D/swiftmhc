from math import sqrt
import torch
from Bio.PDB.PDBParser import PDBParser
from swiftmhc.config import config
from swiftmhc.modules.cross_structure_module import CrossStructureModule
from swiftmhc.tools.pdb import get_atom14_positions


pdb_parser = PDBParser()


def test_omega_calculation():
    m = CrossStructureModule(config)

    pdb = pdb_parser.get_structure("BA-55224", "tests/data/BA-55224.pdb")
    residues = list(pdb[0]["M"].get_residues())

    xyz = []
    for residue in residues:
        pos, mask = get_atom14_positions(residue)
        xyz.append(pos)

    xyz = torch.stack(xyz)

    mask = torch.ones(xyz.shape[:-2])

    omegas = m._calculate_omegas_from_positions(xyz, mask)

    for index in range(omegas.shape[0]):
        if residues[index + 1].get_resname() == "PRO":
            continue

        sin, cos = omegas[index]

        l = sqrt(sin * sin + cos * cos)
        sin = sin / l
        cos = cos / l

        # omega must be close to pi radials, 180 degrees
        assert cos < -0.5, (
            f" between {residues[index]} and {residues[index + 1]}: sin,cos={sin.item()}, {cos.item()}"
        )
