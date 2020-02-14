#!/usr/bin/env python3

""" Simulate 3D aggregate mixing

Generate simulated aggregates that can be analyzed by `analyze_3d_aggregate_mixing.py`

.. code-block:: bash

    $ ./simulate_3d_aggregate_mixing.py \\
        --num-red 400 \\
        --num-green 127 \\
        --aggregate-radius 75.3 \\
        --neighbor-radius 20 \\
        --same-cell-radius 5 \\
        --num-batches 16 \\
        ../data/sim_uniform_pos

Where the options are:

* ``num_red``: Number of "red" (mKate) cells to generate
* ``num_green``: Number of "green" (GFP) cells to generate
* ``aggregate_radius``: um - Radius of the spherical aggregate
* ``neighbor_radius``: um - Cells this close or closer are "neighbors"
* ``same_cell_radius``: um - Cells this close or closer are "the same cell"
* ``num_batches``: Number of aggregates to simulate (to match the number of empirical samples)

"""

import sys
import shutil
import pathlib
import argparse
from typing import Optional

# Allow the scripts directory to be used in-place
THISDIR = pathlib.Path(__file__).resolve().parent
BASEDIR = THISDIR.parent
if THISDIR.name == 'scripts' and (BASEDIR / 'cm_microtissue_struct').is_dir():
    sys.path.insert(0, str(BASEDIR))

# Our own imports
from cm_microtissue_struct.simulation import (
    simulate_spheres_in_sphere, split_red_green
)
from cm_microtissue_struct.io import save_position_data
from cm_microtissue_struct.plotting import plot_3d_sphere_cloud
from cm_microtissue_struct.consts import (
    NUM_RED, NUM_GREEN, AGGREGATE_RADIUS, SAME_CELL_RADIUS, NEIGHBOR_RADIUS,
)


# Main function


def simulate_mixing(outdir: pathlib.Path,
                    prefix: Optional[str] = None,
                    distribution: str = 'uniform',
                    num_red: int = NUM_RED,
                    num_green: int = NUM_GREEN,
                    num_batches: int = 1,
                    aggregate_radius: float = AGGREGATE_RADIUS,
                    neighbor_radius: float = NEIGHBOR_RADIUS,
                    same_cell_radius: float = SAME_CELL_RADIUS):
    """ Simulate mixing two populations in an aggregate

    :param Path outdir:
        The base directory to write results to
    :param str prefix:
        The prefix for each simulation
    :param str distribution:
        The distribution to simulate
    :param int num_red:
        The number of "red" cells to generate
    :param int num_green:
        The number of "green" cells to generate
    :param int num_batches:
        The number of times to run the simulation
    :param float aggregate_radius:
        The radius for the overall aggregate
    :param float neighbor_radius:
        (UNUSED) Cells closer than this are "neighbors"
    :param float same_cell_radius:
        Cells closer than this are the "same cell" (hard shell model)
    """

    if outdir.is_dir():
        shutil.rmtree(str(outdir))
    outdir.mkdir(parents=True)

    if prefix is None:
        prefix = distribution

    for batch_id in range(1, num_batches+1):
        print(f'Simulating batch {batch_id} of {num_batches}')
        green_dir = outdir / f'{prefix}{batch_id:02d}_gfp_statistics'
        green_file = green_dir / f'{prefix}{batch_id:02d}_gfp_Position.csv'

        red_dir = outdir / f'{prefix}{batch_id:02d}_spot_statistics'
        red_file = red_dir / f'{prefix}{batch_id:02d}_spot_Position.csv'

        plotfile = outdir / f'{prefix}{batch_id:02d}_sphere_plot.png'

        # Make a mixed sphere with cells uniformly distributed
        print(f'Simulating {num_red + num_green} spheres')
        all_points = simulate_spheres_in_sphere(num_red + num_green,
                                                particle_radius=same_cell_radius,
                                                sphere_radius=aggregate_radius,
                                                umin=0.0, umax=1.0, udist='uniform')

        # Assign cells to be red or green using a selected distribution
        red_points, green_points = split_red_green(all_points,
                                                   num_red=num_red,
                                                   num_green=num_green,
                                                   udist=distribution)

        plot_3d_sphere_cloud([red_points, green_points], ['red', 'green'],
                             radii=same_cell_radius, figsize=(16, 16),
                             outfile=plotfile)
        save_position_data(green_file, green_points)
        save_position_data(red_file, all_points)


# Command line interface


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', help='Prefix for the directories')
    parser.add_argument("--num-red", type=int, default=NUM_RED,
                        help='Number of "red" (mKate) cells to generate')
    parser.add_argument("--num-green", type=int, default=NUM_GREEN,
                        help='Number of "green" (GFP) cells to generate')
    parser.add_argument('--num-batches', type=int, default=3,
                        help='Number of simulations to run')
    parser.add_argument("--aggregate-radius", type=float, default=AGGREGATE_RADIUS,
                        help='Radius of the spherical aggregate in um')
    parser.add_argument("--neighbor-radius", type=float, default=NEIGHBOR_RADIUS,
                        help='Cells this close or closer are "neighbors"')
    parser.add_argument("--same-cell-radius", type=float, default=SAME_CELL_RADIUS,
                        help='Cells this close or closer are "the same cell"')
    parser.add_argument('-d', '--distribution', default='uniform',
                        choices=('uniform', 'left_triangle', 'right_triangle',
                                 'inside', 'outside'))
    parser.add_argument('outdir', type=pathlib.Path,
                        help='Directory to write the plots out to')
    return parser.parse_args(args=args)


def main(args=None):
    args = parse_args(args=args)
    simulate_mixing(**vars(args))


if __name__ == '__main__':
    main()
