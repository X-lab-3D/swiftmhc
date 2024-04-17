from importlib import resources

from typing import Dict, Tuple, List

import torch
from openmm.app.topology import Topology
from openmm.app.modeller import Modeller
from openmm.app import PDBFile, NoCutoff, Simulation, PDBReporter, StateDataReporter, ForceField, HBonds
from openmm.unit import *
from openmm import LangevinIntegrator, Platform, Vec3
from openmm.app.element import Element

from openfold.np.residue_constants import restype_name_to_atom14_names, restypes, restype_1to3, residue_atoms, rigid_group_atom_positions


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
    atom_nr = 0
    for chain_id, residue_numbers, aatype, atom14_positions, atom14_mask in chain_data:

        chain = topology.addChain(chain_id)

        for residue_index, amino_acid_index in enumerate(aatype):

            # check for at least a C alpha
            if not atom14_mask[residue_index, 1]:
                prev_c = None
                continue

            amino_acid_code = amino_acid_order[amino_acid_index]
            bonds = bonds_per_amino_acid[amino_acid_code]

            residue = topology.addResidue(amino_acid_code, chain, residue_numbers[residue_index])

            atoms_by_name = {}
            positions_by_name = {}
            for atom_index, atom_name in enumerate(restype_name_to_atom14_names[amino_acid_code]):
                if atom14_mask[residue_index, atom_index]:

                    coords = atom14_positions[residue_index, atom_index]
                    pos = 0.1 * Vec3(coords[0].item(), coords[1].item(), coords[2].item())
                    positions.append(pos)

                    atom_nr += 1
                    atom = topology.addAtom(atom_name, Element.getBySymbol(atom_name[0]), residue, str(atom_nr))

                    atoms_by_name[atom_name] = atom
                    positions_by_name[atom_name] = atom14_positions[residue_index, atom_index]

            if 'C' in atoms_by_name:
                prev_c = atoms_by_name['C']
            else:
                prev_c = None

            # C-terminus:
            if 'CA' in positions_by_name and 'C' in positions_by_name and 'O' in positions_by_name and \
               ((residue_index + 1) >= len(aatype) or not atom14_mask[residue_index + 1][0]):

                # compute the terminal oxygen, from the other oxygen
                ca_pos = positions_by_name['CA']
                c_pos = positions_by_name['C']
                o_pos = positions_by_name['O']

                ca_c = c_pos - ca_pos
                c_o = o_pos - c_pos

                c_o_norm = torch.linalg.vector_norm(c_o)
                c_o_unit = c_o / c_o_norm
                ca_c_unit = ca_c / torch.linalg.vector_norm(ca_c)

                # projection
                t = c_o_norm * ca_c_unit * torch.linalg.vecdot(ca_c_unit, c_o_unit)
                n = c_o - t

                # mirror the c-o bond
                oxt_pos = o_pos - 2 * n

                # tell OpenMM
                atom_nr += 1
                oxt = topology.addAtom("OXT", Element.getBySymbol("O"), residue, str(atom_nr))
                pos = 0.1 * Vec3(oxt_pos[0].item(), oxt_pos[1].item(), oxt_pos[2].item())
                positions.append(pos)

    positions = positions * nanometers

    topology.createStandardBonds()
    topology.createDisulfideBonds(positions)

    return Modeller(topology, positions)


def minimize(modeller: Modeller):
    """
    Do OpenMM energy minimization on the input structure modeller.
    """

    chain_ids = [chain.id for chain in modeller.topology.chains()]

    if torch.cuda.is_available():
        platform = Platform.getPlatformByName("CUDA")
    else:
        platform = Platform.getPlatformByName("CPU")

    forcefield = ForceField('amber99sb.xml', 'tip3p.xml')

    modeller.addHydrogens(forcefield, pH=7.0)

    system = forcefield.createSystem(modeller.topology, nonbondedMethod=NoCutoff, nonbondedCutoff=1.0 * nanometer)

    integrator = LangevinIntegrator(300 * kelvin, 1.0 / picosecond, 2.0 * femtosecond)

    simulation = Simulation(modeller.topology, system, integrator, platform)

    simulation.context.setPositions(modeller.positions)

    simulation.minimizeEnergy()

    # preserve chain ids
    for chain, id_ in zip(modeller.topology.chains(), chain_ids):
        chain.id = id_

