#!/usr/bin/python

import os
import logging
import sys
from argparse import ArgumentParser
from time import time

from openmm.app import PDBFile, NoCutoff, Simulation, PDBReporter, StateDataReporter, ForceField, HBonds
from openmm.unit import picosecond, femtosecond, kelvin, nanometer, md_unit_system
from openmm import LangevinIntegrator


arg_parser = ArgumentParser()
arg_parser.add_argument("pdb_list", help="a file listing the pdb files to process")


_log = logging.getLogger(__name__)


if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    args = arg_parser.parse_args()

    for path in open(args.pdb_list, 'r').read().strip().split('\n'):

        name = os.path.splitext(os.path.basename(path))[0]

        pdb = PDBFile(path)

        forcefield = ForceField('amber99sb.xml', 'tip3p.xml')
        system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff, nonbondedCutoff=1.0 * nanometer)

        integrator = LangevinIntegrator(300 * kelvin, 1.0 / picosecond, 2.0 * femtosecond)

        simulation = Simulation(pdb.topology, system, integrator)

        simulation.context.setPositions(pdb.positions)

        state = simulation.context.getState(getEnergy=True, getPositions=True)
        energy = state.getPotentialEnergy().value_in_unit_system(md_unit_system)
        _log.info(f"Initial Potential Energy: {energy:10.3f}")

        t0 = time()

        simulation.minimizeEnergy()

        t1 = time()
        _log.info(f"minimized in {t1 - t0} seconds")

        state = simulation.context.getState(getEnergy=True, getPositions=True)
        energy = state.getPotentialEnergy().value_in_unit_system(md_unit_system)
        _log.info(f"Final Potential Energy: {energy:10.3f}")

        with open(f"minimized-{name}.pdb", 'w') as pdb_file:
            PDBFile.writeFile(pdb.topology, state.getPositions(), pdb_file)
