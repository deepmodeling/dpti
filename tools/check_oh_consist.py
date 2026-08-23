#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import numpy as np

# Direct execution sets sys.path to tools/, so add the repository for source-tree use.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dpti.lib import dump, water


def get_oh_distance_stats(dump_path):
    """Return per-frame O-H distance statistics from a LAMMPS dump trajectory."""
    trajectories = dump.split_traj(Path(dump_path).read_text().splitlines())
    if not trajectories:
        raise ValueError(f"no LAMMPS trajectory frames found in {dump_path}")

    bounds, tilt = dump.get_dumpbox(trajectories[0])
    _, box = dump.dumpbox2box(bounds, tilt)
    atom_types = dump.get_atype(trajectories[0])
    positions = dump.get_posi(trajectories[0])
    oh_list = water.min_oh_list(box, atom_types, positions)

    stats = []
    for index, trajectory in enumerate(trajectories):
        bounds, tilt = dump.get_dumpbox(trajectory)
        _, box = dump.dumpbox2box(bounds, tilt)
        positions = dump.get_posi(trajectory)
        distances = water.dist_via_oh_list(box, positions, oh_list)
        stats.append(
            (index, np.min(distances), np.max(distances), np.average(distances))
        )
    return stats


def main(argv=None):
    """Parse command-line arguments and print O-H distance statistics."""
    parser = argparse.ArgumentParser(
        description="Check O-H bond consistency across a LAMMPS dump trajectory"
    )
    parser.add_argument("DUMP", help="LAMMPS dump trajectory to inspect")
    args = parser.parse_args(argv)

    for values in get_oh_distance_stats(args.DUMP):
        print(*values)


if __name__ == "__main__":
    main()
