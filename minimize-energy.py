#!/usr/bin/env python

import os
import logging
import sys
from argparse import ArgumentParser
from time import time
from multiprocessing import Pool

from openmm.app import PDBFile, NoCutoff, Simulation, PDBReporter, StateDataReporter, ForceField, HBonds
from openmm.unit import picosecond, femtosecond, kelvin, nanometer, md_unit_system
from openmm import LangevinIntegrator


arg_parser = ArgumentParser()
arg_parser.add_argument("pdb_list", help="a file listing the pdb files to process")
arg_parser.add_argument("process_count", type=int)


_log = logging.getLogger(__name__)


def minimize(input_path: str):

    directory_path = os.path.dirname(input_path)
    name = os.path.splitext(os.path.basename(input_path))[0]

    pdb = PDBFile(input_path)

    forcefield = ForceField('amber99sb.xml', 'tip3p.xml')
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff, nonbondedCutoff=1.0 * nanometer)

    integrator = LangevinIntegrator(300 * kelvin, 1.0 / picosecond, 2.0 * femtosecond)

    simulation = Simulation(pdb.topology, system, integrator)

    simulation.context.setPositions(pdb.positions)

    state = simulation.context.getState(getEnergy=True, getPositions=True)
    energy = state.getPotentialEnergy().value_in_unit_system(md_unit_system)
    _log.info(f"{input_path} initial potential energy: {energy:10.3f}")

    t0 = time()
    simulation.minimizeEnergy()
    t1 = time()
    _log.info(f"{input_path} minimized in {t1 - t0} seconds")

    state = simulation.context.getState(getEnergy=True, getPositions=True)
    energy = state.getPotentialEnergy().value_in_unit_system(md_unit_system)
    _log.info(f"{input_path} final potential energy: {energy:10.3f}")

    with open(os.path.join(directory_path, f"minimized-{name}.pdb"), 'w') as pdb_file:
        PDBFile.writeFile(pdb.topology, state.getPositions(), pdb_file)


if __name__ == "__main__":

    logging.basicConfig(filename="energy-minimize.log", filemode='a', level=logging.INFO)

    args = arg_parser.parse_args()

    paths = open(args.pdb_list, 'r').read().strip().split('\n')

    with Pool(args.process_count) as pool:
        pool.map(minimize, paths)
