#!/usr/bin/env python3

""" Stats for 3d aggregate mixing

Using the results of ``analyze_3d_aggregate_mixing`` and ``simulate_3d_aggregate_mixing``,
generate curves that compare empirical and theoretical distributions of cells in 3D.

To generate the comparison to the paper:

.. code-block:: bash

    $ ./stats_3d_aggregate_mixing.py \\
        ../data/empirical_pos \\
        ../data/sim_uniform_pos

Where ``../data/empirical_pos`` is the path to the empirical data and
``../data/sim_uniform_pos`` is the path to the simulated data.

Final plots will be written out under ``../data/sim_uniform_pos``

"""

# Imports
import sys
import pathlib
import argparse
from typing import Optional

# Allow the scripts directory to be used in-place
THISDIR = pathlib.Path(__file__).resolve().parent
BASEDIR = THISDIR.parent
if THISDIR.name == 'scripts' and (BASEDIR / 'cm_microtissue_struct').is_dir():
    sys.path.insert(0, str(BASEDIR))

# 3rd party
import numpy as np

from scipy.stats import ks_2samp

import matplotlib.pyplot as plt

# Our own imports
from cm_microtissue_struct.io import load_summary_datadir
from cm_microtissue_struct.stats import bin_cm_counts, bin_any_counts
from cm_microtissue_struct.plotting import set_plot_style, add_lineplot

# Constants

PLOT_STYLE = 'light'

# Main function


def stats_for_mixing(datadir: pathlib.Path,
                     simdir: pathlib.Path,
                     outdir: Optional[pathlib.Path] = None,
                     plot_style: str = PLOT_STYLE):
    """ Calculate stats for mixing

    :param Path datadir:
        Directory containing analyzed empirical data
    :param Path simdir:
        Directory containing analyzed simulated data
    :param Path outdir:
        If not None, where to write plots and stats and etc
    """
    if plot_style.startswith('dark'):
        sim_palette = 'wheel_greywhite'
    else:
        sim_palette = 'Greys'

    exp_cm_df, exp_cf_df, exp_any_df, exp_volume_df = load_summary_datadir(datadir)
    print(exp_volume_df.head())

    sim_cm_df, sim_cf_df, sim_any_df, sim_volume_df = load_summary_datadir(simdir)
    print(sim_volume_df.head())

    # Subset the simulated and real data
    exp_cm_mask = np.logical_or(exp_cm_df['FibSource'] == 'fetal',
                                exp_cm_df['FibSource'] == 'adult')
    exp_cm_df = exp_cm_df[exp_cm_mask]

    exp_any_mask = np.logical_or(exp_any_df['FibSource'] == 'fetal',
                                 exp_any_df['FibSource'] == 'adult')
    exp_any_df = exp_any_df[exp_any_mask]

    # Subset the simulated and real data
    exp_volume_mask = np.logical_or(exp_volume_df['FibSource'] == 'fetal',
                                    exp_volume_df['FibSource'] == 'adult')
    exp_volume_df = exp_volume_df[exp_volume_mask]

    print('Mean CMs: {}'.format(exp_volume_df['NumCMs'].mean()))
    print('Mean CFs: {}'.format(exp_volume_df['NumCFs'].mean()))
    print('Mean Radius: {}'.format(exp_volume_df['SphereRadius'].mean()))
    print('')

    sim_cm_mask = sim_cm_df['FibSource'] == 'uniform'
    sim_cm_df = sim_cm_df[sim_cm_mask]

    sim_any_mask = sim_any_df['FibSource'] == 'uniform'
    sim_any_df = sim_any_df[sim_any_mask]

    print('Got {} experimental samples'.format(exp_cm_df.shape[0]))
    print('Got {} simulated samples'.format(sim_cm_df.shape[0]))

    # Use the any simulations to look at edge bias
    exp_any_sample = exp_any_df['Dist All to Center'].values
    sim_any_sample = sim_any_df['Dist All to Center'].values
    res = ks_2samp(exp_any_sample, sim_any_sample)
    print('Dist All to Center: ks={}, p={}'.format(res.statistic, res.pvalue))

    max_r = 100

    exp_any_counts = bin_any_counts(exp_any_df, max_r=max_r)
    sim_any_counts = bin_any_counts(sim_any_df, max_r=max_r)

    # Plot
    xlim = [0, max_r]
    ylim = [0, 220]
    outfile = simdir / 'dist_to_center_final.svg'
    with set_plot_style(plot_style) as style:
        fig, ax1 = plt.subplots(1, 1, figsize=(1.6*5, 1.2*5))

        add_lineplot(ax=ax1, data=exp_any_counts,
                     x='Dist to Center', y='Count Dist to Center',
                     palette='Blues', label='Measured')
        add_lineplot(ax=ax1, data=sim_any_counts,
                     x='Dist to Center', y='Count Dist to Center',
                     palette=sim_palette, label='Simulated')
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        ax1.set_ylabel('# Cells')
        ax1.set_xlabel('Distance to Center ($\\mu m$)')
        ax1.set_title('Counts of cells by distance to aggregate center')
        ax1.legend()

        style.show(outfile=outfile)

    exp_cm_sample = exp_cm_df['Num CM Near CM'].values
    sim_cm_sample = sim_cm_df['Num CM Near CM'].values
    res = ks_2samp(exp_cm_sample, sim_cm_sample)
    print('CM Near CM: ks={}, p={}'.format(res.statistic, res.pvalue))

    exp_cf_sample = exp_cm_df['Num CF Near CM'].values
    sim_cf_sample = sim_cm_df['Num CF Near CM'].values
    res = ks_2samp(exp_cf_sample, sim_cf_sample)
    print('CF Near CM: ks={}, p={}'.format(res.statistic, res.pvalue))

    # Convert to counts
    exp_cm_counts = bin_cm_counts(exp_cm_df)
    sim_cm_counts = bin_cm_counts(sim_cm_df)

    # Plot
    xlim = [0, 20]
    ylim = [0, 140]
    outfile = simdir / 'dist_final.svg'
    with set_plot_style(plot_style) as style:

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(1.4*5, 2*5))

        add_lineplot(ax=ax1, data=exp_cm_counts,
                     x='Num CM Near CM', y='Count CM Near CM',
                     palette='Blues', label='Measured')
        add_lineplot(ax=ax1, data=sim_cm_counts,
                     x='Num CM Near CM', y='Count CM Near CM',
                     palette=sim_palette, label='Simulated')
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        ax1.set_ylabel('# CM')
        ax1.set_xlabel('# Nearest CM Neighbors')
        ax1.set_title('CM near CM')
        ax1.legend()

        add_lineplot(ax=ax2, data=exp_cm_counts,
                     x='Num CF Near CM', y='Count CF Near CM',
                     palette='Blues', label='Measured')
        add_lineplot(ax=ax2, data=sim_cm_counts,
                     x='Num CF Near CM', y='Count CF Near CM',
                     palette=sim_palette, label='Simulated')
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)
        ax2.set_ylabel('# CM')
        ax2.set_xlabel('# Nearest CF Neighbors')
        ax2.set_title('CF near CM')
        ax2.legend()

        style.show(outfile=outfile)

    xlim = [0, 20]
    ylim = [0, 35]
    outfile = simdir / 'dist_pct_final.svg'
    with set_plot_style(plot_style) as style:

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(1.4*5, 2*5))

        add_lineplot(ax=ax1, data=exp_cm_counts,
                     x='Num CM Near CM', y='Percent CM Near CM',
                     palette='Blues', label='Measured')
        add_lineplot(ax=ax1, data=sim_cm_counts,
                     x='Num CM Near CM', y='Percent CM Near CM',
                     palette=sim_palette, label='Simulated')
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        ax1.set_ylabel('% CM')
        ax1.set_xlabel('# Nearest CM Neighbors')
        ax1.set_title('CM near CM')
        ax1.legend()

        add_lineplot(ax=ax2, data=exp_cm_counts,
                     x='Num CF Near CM', y='Percent CF Near CM',
                     palette='Blues', label='Measured')
        add_lineplot(ax=ax2, data=sim_cm_counts,
                     x='Num CF Near CM', y='Percent CF Near CM',
                     palette=sim_palette, label='Simulated')
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)
        ax2.set_ylabel('% CM')
        ax2.set_xlabel('# Nearest CF Neighbors')
        ax2.set_title('CF near CM')
        ax2.legend()

        style.show(outfile=outfile)


# Command line interface


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-style', default=PLOT_STYLE)
    parser.add_argument('datadir', type=pathlib.Path,
                        help='Directory containing analyzed empirical data')
    parser.add_argument('simdir', type=pathlib.Path,
                        help='Directory containing analyzed simulated data')
    return parser.parse_args(args=args)


def main(args=None):
    args = parse_args(args=args)
    stats_for_mixing(**vars(args))


if __name__ == '__main__':
    main()
