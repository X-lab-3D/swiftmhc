#!/usr/bin/env python

from argparse import ArgumentParser

import pdbfixer
from openmm import app
from openmm.app.pdbfile import PDBFile
import openmm
from openmm.unit import *


arg_parser = ArgumentParser()
arg_parser.add_argument("input_file")
arg_parser.add_argument("output_file")


if __name__ == "__main__":

    args = arg_parser.parse_args()

    platform = openmm.Platform.getPlatformByName("CUDA")

    structure = pdbfixer.PDBFixer(filename=args.input_file)
    structure.findMissingResidues()
    structure.findNonstandardResidues()
    structure.replaceNonstandardResidues()
    structure.findMissingAtoms()
    structure.addMissingAtoms()
    structure.addMissingHydrogens(pH=7.0)

    forcefield = app.ForceField('amber99sb.xml', 'tip3p.xml')
    system = forcefield.createSystem(structure.topology, nonbondedMethod=app.NoCutoff, nonbondedCutoff=1.0 * nanometer)

    integrator = openmm.LangevinIntegrator(310 * kelvin, 1.0 / picosecond, 2.0 * femtosecond)

    simulation = app.Simulation(structure.topology, system, integrator, platform)

    simulation.context.setPositions(structure.positions)

    state = simulation.context.getState(getEnergy=True, getPositions=True)

    energy_before = state.getPotentialEnergy().value_in_unit_system(md_unit_system)

    simulation.minimizeEnergy()

    state = simulation.context.getState(getEnergy=True, getPositions=True)

    energy_after = state.getPotentialEnergy().value_in_unit_system(md_unit_system)

    with open(args.output_file, 'wt') as output_file:
        PDBFile.writeFile(structure.topology, state.getPositions(), output_file)
