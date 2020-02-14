
""" Analyze lightsheet 3D aggregate distributions

Classes:

* :py:class:`SpotData`: Store the spot records for a single sample

Functions:

* :py:func:`group_directories`: Collect all samples under a single directory and
    group them into :py:class:`SpotData` objects.

"""

import re
import shutil
import pathlib
from typing import Optional, List

# 3rd party
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

# Our own imports
from .consts import NEIGHBOR_RADIUS, SAME_CELL_RADIUS
from .io import load_position_data
from .simulation import (
    concat_points, mask_points, count_neighbors, nearest_neighbors,
)
from .plotting import set_plot_style, add_histogram
from . import poly

# Constants
FIGSIZE = (16, 16)
PLOT_STYLE = 'light'
SUFFIX = '.svg'

GROUP_TYPE = 'split_label'

MARKERSIZE = 10

reGFP_DIR = re.compile(r'^(?P<prefix>.*)_gfp.?_statistics$', re.IGNORECASE)
reMKATE_DIR = re.compile(r'^(?P<prefix>.*)_spot.?_statistics$', re.IGNORECASE)
reCM_DIR = re.compile(r'^(?P<prefix>.*)_cm.?_statistics$', re.IGNORECASE)
reCF_DIR = re.compile(r'^(?P<prefix>.*)_cf.?_statistics$', re.IGNORECASE)

# Classes


class SpotData(object):
    """ Store the spots for a single measurement

    :param str prefix:
        The prefix for this aggregate directory
    :param Path gfp_dir:
        The directory containing GFP coordinates
    :param Path mkate_dir:
        The directory containing mKate coordinates
    :param str group_type:
        How are the points grouped ('split_label' or 'double_label')
    :param float neighbor_radius:
        Maximum distance for two cells to be "neighbors"
    :param float same_cell_radius:
        Maximum distance for two cells to be "neighbors"
    :param str plot_style:
        Style to use for plots
    :param Path outdir:
        If not None, folder to write the files to
    :param str suffix:
        Suffix to save the plot files with
    """

    def __init__(self,
                 prefix: str,
                 gfp_dir: pathlib.Path,
                 mkate_dir: pathlib.Path,
                 group_type: str = GROUP_TYPE,
                 neighbor_radius: float = NEIGHBOR_RADIUS,
                 same_cell_radius: float = SAME_CELL_RADIUS,
                 outdir: Optional[pathlib.Path] = None,
                 plot_style: str = PLOT_STYLE,
                 suffix: str = SUFFIX):
        self.prefix = prefix
        self.gfp_dir = gfp_dir
        self.mkate_dir = mkate_dir

        self.group_type = group_type

        if outdir is None:
            outdir = self.gfp_dir.parent / f'{self.prefix}_plots_{plot_style}'
        self.outdir = outdir

        self.neighbor_radius = neighbor_radius
        self.same_cell_radius = same_cell_radius

        self.gfp_points = None
        self.mkate_points = None

        self.red_points = None
        self.green_points = None

        # Plot styles
        self.plot_style = plot_style
        self.suffix = suffix

        # Plot bounds
        self.num_cells_min = 0
        self.num_cells_max = 20

        self.dist_cells_min = 0
        self.dist_cells_max = 30

    def __repr__(self):
        return 'SpotData({})'.format(self.prefix)

    def load_points(self, type: str):
        """ Load the points for each input data type

        :param str type:
            The type of point (either gfp or mkate)
        """
        indir = getattr(self, '{}_dir'.format(type))
        pointfile = None
        for datafile in indir.iterdir():
            if datafile.name.endswith('_Cell_Position.csv'):
                pointfile = datafile
                break
            elif datafile.name.endswith('_Position.csv'):
                pointfile = datafile
                break
        print('Loading {}'.format(pointfile))
        setattr(self, '{}_points'.format(type), load_position_data(pointfile))

    def group_points(self):
        """ Group the points multiple ways """
        getattr(self, 'group_points_{}'.format(self.group_type))()

    def group_points_split_label(self):
        """ Convert GFP+/mKate+ to red/green """

        print('Got {} initial EGFP points'.format(len(self.gfp_points[0])))
        print('Got {} initial mkate points'.format(len(self.mkate_points[0])))

        self.green_points = self.gfp_points
        self.red_points = self.mkate_points

        print('Got {} green'.format(len(self.green_points[0])))
        print('Got {} red'.format(len(self.red_points[0])))

    def group_points_double_green(self):
        """ Convert GFP+/mKate+ to red/green

        All cells are labeled in red. Some cells are also labeled in green
        """

        print('Got {} initial EGFP points'.format(len(self.gfp_points[0])))
        print('Got {} initial mkate points'.format(len(self.mkate_points[0])))

        dist_green_to_red = nearest_neighbors(self.mkate_points, self.gfp_points)
        green_mask = dist_green_to_red <= self.same_cell_radius

        self.green_points = mask_points(self.mkate_points, green_mask)
        self.red_points = mask_points(self.mkate_points, ~green_mask)

        print('Assigned {}'.format(np.sum(green_mask)))
        print('Got {} green'.format(len(self.green_points[0])))
        print('Got {} red'.format(len(self.red_points[0])))

    def group_points_double_red(self):
        """ Convert GFP+/mKate+ to red/green

        All cells are labeled in green. Some cells are also labeled in red
        """

        print('Got {} initial EGFP points'.format(len(self.gfp_points[0])))
        print('Got {} initial mkate points'.format(len(self.mkate_points[0])))

        dist_red_to_green = nearest_neighbors(self.gfp_points, self.mkate_points)
        red_mask = dist_red_to_green <= self.same_cell_radius

        self.red_points = mask_points(self.gfp_points, red_mask)
        self.green_points = mask_points(self.gfp_points, ~red_mask)

        print('Assigned {}'.format(np.sum(red_mask)))
        print('Got {} green'.format(len(self.green_points[0])))
        print('Got {} red'.format(len(self.red_points[0])))

    def make_outdir(self):
        """ Make the output directory """
        outdir = self.outdir
        if outdir.is_dir():
            shutil.rmtree(str(outdir))
        outdir.mkdir(exist_ok=True, parents=True)

    def calc_volume_stats(self):
        """ Calculate stats for the volume as a whole """

        all_points = np.concatenate([
            np.stack(self.red_points, axis=1),
            np.stack(self.green_points, axis=1),
        ], axis=0)
        vertex_points = poly.vertices_of_polyhedron(all_points)

        stats = {}

        # Calculate some stats
        volume = poly.volume_of_polyhedron(all_points)
        stats['Volume'] = volume
        centroid = poly.centroid_of_polyhedron(all_points)

        stats['Center X'] = centroid[0]
        stats['Center Y'] = centroid[1]
        stats['Center Z'] = centroid[2]

        radius = np.sqrt(np.sum((vertex_points - centroid[np.newaxis, :])**2, axis=1))
        stats['MinRadius'] = np.min(radius)
        stats['MaxRadius'] = np.max(radius)
        stats['MeanRadius'] = np.mean(radius)

        # 4/3*pi*r**3 = V
        # (3/4*V/pi)**(1/3) = r
        sphere_radius = (volume * 3/4 / np.pi)**(1/3)
        stats['SphereRadius'] = sphere_radius

        stats['NumCMs'] = self.red_points[0].shape[0]
        stats['NumCFs'] = self.green_points[0].shape[0]
        stats['NumTotal'] = all_points.shape[0]

        outfile = self.outdir / '{}_volume_stats.xlsx'.format(self.prefix)

        df = pd.DataFrame({k: [v] for k, v in stats.items()})
        print('Writing: {}'.format(outfile))
        df.to_excel(outfile)

    def calc_point_stats(self):
        """ Make some plots that are cool """
        outdir = self.outdir
        plot_style = self.plot_style
        suffix = self.suffix
        radius = self.neighbor_radius

        # Plot boundaries for histograms
        dist_min, dist_max = self.dist_cells_min, self.dist_cells_max
        num_min, num_max = self.num_cells_min, self.num_cells_max

        collection_name = self.prefix

        red_points = self.red_points
        green_points = self.green_points

        # Red-centric stats
        with set_plot_style(plot_style) as style:
            fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
            ax0, ax1, ax2, ax3 = axes.ravel()

            red_to_green_dists = nearest_neighbors(red_points, green_points)
            red_to_red_dists = nearest_neighbors(red_points, red_points, num_closest=2)

            red_to_num_green_neighbors = count_neighbors(red_points, green_points, radius=radius)
            red_to_num_red_neighbors = count_neighbors(red_points, red_points, radius=radius) - 1

            add_histogram(ax0, red_to_green_dists,
                          title=collection_name,
                          range=(dist_min, dist_max),
                          xlabel='Distance red to green cell')
            add_histogram(ax1, red_to_red_dists,
                          title=collection_name,
                          range=(dist_min, dist_max),
                          xlabel='Distance red to red cell')
            add_histogram(ax2, red_to_num_green_neighbors,
                          title=collection_name,
                          range=(num_min, num_max),
                          bins=(num_max - num_min),
                          kernel_bandwidth=1,
                          xlabel='Number of green neighbors for red')
            add_histogram(ax3, red_to_num_red_neighbors,
                          title=collection_name,
                          range=(num_min, num_max),
                          bins=(num_max - num_min),
                          kernel_bandwidth=1,
                          xlabel='Number of red neighbors for red')
            outfile = outdir / '{}_red_stats{}'.format(collection_name, suffix)
            style.show(outfile=outfile)

        # Green-centric stats
        with set_plot_style(plot_style) as style:
            fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
            ax0, ax1, ax2, ax3 = axes.ravel()

            green_to_red_dists = nearest_neighbors(green_points, red_points)
            green_to_green_dists = nearest_neighbors(green_points, green_points, num_closest=2)

            green_to_num_red_neighbors = count_neighbors(green_points, red_points, radius=radius)
            green_to_num_green_neighbors = count_neighbors(green_points, green_points, radius=radius) - 1

            add_histogram(ax0, green_to_green_dists,
                          title=collection_name,
                          range=(dist_min, dist_max),
                          xlabel='Distance green to green cell')
            add_histogram(ax1, green_to_red_dists,
                          title=collection_name,
                          range=(dist_min, dist_max),
                          xlabel='Distance green to red cell')
            add_histogram(ax2, green_to_num_green_neighbors,
                          title=collection_name,
                          range=(num_min, num_max),
                          bins=(num_max - num_min),
                          kernel_bandwidth=1,
                          xlabel='Number of green neighbors for green')
            add_histogram(ax3, green_to_num_red_neighbors,
                          title=collection_name,
                          range=(num_min, num_max),
                          bins=(num_max - num_min),
                          kernel_bandwidth=1,
                          xlabel='Number of red neighbors for green')
            outfile = outdir / '{}_green_stats{}'.format(collection_name, suffix)
            style.show(outfile=outfile)

        # Compare distributions
        label = ['Red to Red' for _ in red_to_red_dists] + \
                ['Red to Green' for _ in red_to_green_dists] + \
                ['Green to Green' for _ in green_to_green_dists] + \
                ['Green to Red' for _ in green_to_red_dists]
        source = ['Red' for _ in red_to_red_dists] + \
                 ['Red' for _ in red_to_green_dists] + \
                 ['Green' for _ in green_to_green_dists] + \
                 ['Green' for _ in green_to_red_dists]
        target = ['Red' for _ in red_to_red_dists] + \
                 ['Green' for _ in red_to_green_dists] + \
                 ['Green' for _ in green_to_green_dists] + \
                 ['Red' for _ in green_to_red_dists]

        df = pd.DataFrame({
            'Distance': np.concatenate([red_to_red_dists,
                                        red_to_green_dists,
                                        green_to_green_dists,
                                        green_to_red_dists]),
            'Neighbors': np.concatenate([red_to_num_red_neighbors,
                                         red_to_num_green_neighbors,
                                         green_to_num_green_neighbors,
                                         green_to_num_red_neighbors]),
            'Label': label,
            'Source': source,
            'Target': target,
        })
        df.to_excel(str(outdir / '{}_sorted_stats.xlsx'.format(collection_name)))

        # Stash the distances and counts to a file
        print('Num red points: {}'.format(len(red_points[0])))
        print('Num green points: {}'.format(len(green_points[0])))

        points = np.array(concat_points(red_points, green_points))
        center = poly.centroid_of_polyhedron(points.T)

        print('Centroid at {:0.1f} {:0.1f} {:0.1f}'.format(*center))

        dist_all_to_center = np.linalg.norm(points - center[:, np.newaxis], axis=0)
        dist_red_to_center = np.linalg.norm(np.array(red_points) - center[:, np.newaxis], axis=0)
        dist_green_to_center = np.linalg.norm(np.array(green_points) - center[:, np.newaxis], axis=0)

        # Center-of-mass stats
        with set_plot_style(plot_style) as style:
            fig, axes = plt.subplots(1, 3, figsize=(24, 8))
            ax0, ax1, ax2 = axes.ravel()

            # dist_min, dist_max = np.percentile(dists, [5, 95])
            # num_min = np.min(num_neighbors)
            # num_max = np.max(num_neighbors)
            dist_min, dist_max = 0.0, 150.0
            add_histogram(ax0, dist_all_to_center,
                          title=collection_name,
                          range=(dist_min, dist_max),
                          xlabel='Distance all cells to center of the aggregate')
            add_histogram(ax1, dist_red_to_center,
                          title=collection_name,
                          range=(dist_min, dist_max),
                          xlabel='Distance red to center of the aggregate')
            add_histogram(ax2, dist_green_to_center,
                          title=collection_name,
                          range=(dist_min, dist_max),
                          xlabel='Distance green to center of aggregate')
            outfile = outdir / '{}_dist_stats{}'.format(collection_name, suffix)
            style.show(outfile=outfile)

        dist_red_to_red = nearest_neighbors(red_points, red_points, num_closest=2)
        dist_green_to_green = nearest_neighbors(green_points, green_points, num_closest=2)
        dist_red_to_green = nearest_neighbors(red_points, green_points)
        dist_green_to_red = nearest_neighbors(green_points, red_points)

        num_red_to_red = count_neighbors(red_points, red_points, radius=radius) - 1
        num_green_to_green = count_neighbors(green_points, green_points, radius=radius) - 1
        num_red_to_green = count_neighbors(red_points, green_points, radius=radius)
        num_green_to_red = count_neighbors(green_points, red_points, radius=radius)

        dist_all_to_red = nearest_neighbors(points, red_points, num_closest=2)
        dist_all_to_green = nearest_neighbors(points, green_points, num_closest=2)
        num_all_to_red = count_neighbors(points, red_points, radius=radius) - 1
        num_all_to_green = count_neighbors(points, green_points, radius=radius) - 1

        df_all = pd.DataFrame({
            'Dist All to Center': dist_all_to_center,
            'Num CM Near Any': num_all_to_red,
            'Num CF Near Any': num_all_to_green,
            'Dist Any to Nearest CM': dist_all_to_red,
            'Dist Any to Nearest CF': dist_all_to_green,
        })
        df_all.to_excel(str(outdir / '{}_all_stats.xlsx'.format(collection_name)))

        df_red = pd.DataFrame({
            'Dist CM to Center': dist_red_to_center,
            'Dist CM to Nearest CM': dist_red_to_red,
            'Dist CM to Nearest CF': dist_red_to_green,
            'Num CM Near CM': num_red_to_red,
            'Num CF Near CM': num_red_to_green},
        )
        df_red.to_excel(str(outdir / '{}_cm_stats.xlsx'.format(collection_name)))

        df_green = pd.DataFrame({
            'Dist CF to Center': dist_green_to_center,
            'Dist CF to Nearest CM': dist_green_to_red,
            'Dist CF to Nearest CF': dist_green_to_green,
            'Num CM Near CF': num_green_to_red,
            'Num CF Near CF': num_green_to_green},
        )
        df_green.to_excel(str(outdir / '{}_cf_stats.xlsx'.format(collection_name)))

        # Save some final values for plotting
        self.num_red_to_red = num_red_to_red
        self.num_red_to_green = num_red_to_green


# Functions


def group_directories(rootdir: pathlib.Path,
                      group_type: str = GROUP_TYPE,
                      **kwargs) -> List[SpotData]:
    """ Group the directories by GFP vs mKate

    :param Path rootdir:
        Directory to search for analysis directories under
    :returns:
        A list of SpotData objects, with data for both colors
    """
    if group_type == 'split_label':
        re_green_dir = reCF_DIR
        re_red_dir = reCM_DIR
    elif group_type in ('double_green', 'double_red'):
        re_green_dir = reGFP_DIR
        re_red_dir = reMKATE_DIR
    else:
        raise ValueError(f'Unknown group type: "{group_type}"')

    gfp_dirs = {}
    mkate_dirs = {}
    print(f'Loading all dirs under: {rootdir}')

    for subdir in rootdir.iterdir():
        if not subdir.is_dir():
            continue
        match = re_green_dir.match(subdir.name)
        if match:
            print(f'Found GFP dir: {subdir.name}')
            gfp_dirs[match.group('prefix')] = subdir
            continue
        match = re_red_dir.match(subdir.name)
        if match:
            print(f'Found mKate dir: {subdir.name}')
            mkate_dirs[match.group('prefix')] = subdir
            continue

    prefixes = set(gfp_dirs) | set(mkate_dirs)
    missing_pairs = []
    groups = []
    for prefix in prefixes:
        gfp_dir = gfp_dirs.get(prefix)
        if not gfp_dir:
            missing_pairs.append(prefix)
            continue
        mkate_dir = mkate_dirs.get(prefix)
        if not mkate_dir:
            missing_pairs.append(prefix)
            continue
        groups.append(SpotData(prefix, gfp_dir, mkate_dir, group_type=group_type, **kwargs))

    if missing_pairs:
        raise ValueError(f'Got unparied dirs {len(missing_pairs)}: {missing_pairs}')
    return groups
