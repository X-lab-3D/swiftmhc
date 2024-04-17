from importlib import resources

from typing import Dict, Tuple, List

import torch
from openmm.app.topology import Topology
from openmm.app.modeller import Modeller
from openfold.np.residue_constants import restype_name_to_atom14_names, restypes, restype_1to3


amino_acid_order = [restype_1to3[letter] for letter in restypes]


def _load_bond_definitions() -> Dict[str, Tuple[str, str]]:
    """
    Load the interatomic bond definitions from the file.
    """

    stereo_chemical_props = resources.read_text("openfold.resources", "stereo_chemical_props.txt")

    bonds_per_amino_acid = {}
    for line in stereo_chemical_props.splitlines():
        if line.strip() == "-":
            break

        elif len(line.strip()) == 0:
            continue

        bond, amino_acid_code, length, stddev = line.split()
        if bond == "Bond":
            continue  # skip header

        atom1_name, atom2_name = bond.split('-')

        bonds_per_amino_acid[amino_acid_code] = bonds_per_amino_acid.get(amino_acid_code, []) + [(atom1_name, atom2_name)]

    return bonds_per_amino_acid


# load this only once
bonds_per_amino_acid = _load_bond_definitions()


def build_modeller(chain_data: List[Tuple[str,
                                          torch.Tensor,
                                          torch.Tensor,
                                          torch.Tensor,
                                          torch.Tensor]]) -> Modeller:
    """
    Args:
        chain_data: list of chains (id, residue numbers, amino acid type index, atom positions, atom mask)
    Returns:
        OpenMM Modeller object
    """


    topology = Topology()
    positions = []

    prev_c = None
    for chain_id, residue_numbers, aatype, atom14_positions, atom14_mask in chain_data:

        chain = topology.addChain(chain_id)

        for residue_index, amino_acid_index in enumerate(aatype):

            amino_acid_code = amino_acid_order[amino_acid_index]
            bonds = bonds_per_amino_acid[amino_acid_code]

            residue = topology.addResidue(amino_acid_code, chain, residue_numbers[residue_index])

            atoms_by_name = {}
            for atom_index, atom_name in enumerate(restype_name_to_atom14_names[amino_acid_code]):
                if atom14_mask[residue_index, atom_index]:
                    positions.append(atom14_positions[residue_index, atom_index])

                    atom = topology.addAtom(atom_name, atom_name[0], residue)

                    atoms_by_name[atom_name] = atom

            for atom1_name, atom2_name in bonds:
                topology.addBond(atoms_by_name[atom1_name], atoms_by_name[atom2_name])

            if residue_index > 0:
                topology.addBond(atoms_by_name['N'], prev_c)

            prev_c = atoms_by_name['C']

    return Modeller(topology, positions)


def minimize(modeller: Modeller):
    """
    Do OpenMM energy minimization on the input structure modeller.
    """

    if torch.cuda.is_available():
        platform = Platform.getPlatformByName("CUDA")
    else:
        platform = Platform.getPlatformByName("CPU")

    forcefield = ForceField('amber99sb.xml', 'tip3p.xml')
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=NoCutoff, nonbondedCutoff=1.0 * nanometer)

    integrator = LangevinIntegrator(300 * kelvin, 1.0 / picosecond, 2.0 * femtosecond)

    simulation = Simulation(modeller.topology, system, integrator, platform)

    simulation.context.setPositions(modeller.positions)

    simulation.minimizeEnergy()

