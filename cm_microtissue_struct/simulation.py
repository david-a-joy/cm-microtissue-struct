""" Sphere simulation and position tools

Geometry simulation:

* :py:func:`simulate_spheres_in_sphere`: Simulate a random sphere packing of hard
    spheres of identical radius inside a larger sphere

Neighbor counting:

* :py:func:`nearest_neighbors`: Calculate the distance to the nth closest point to
    a given set of points
* :py:func:`count_neighbors`: Calculte the number of points within a radial neighborhood

Point manipulation tools:

* :py:func:`split_red_green`: Split a point list into red/green with a given
    probability distribution
* :py:func:`mask_points`: Subset point lists based on a mask
* :py:func:`concat_points`: Concatenate point lists

"""

# Imports
from typing import Tuple, List

# 3rd party
import numpy as np

from sklearn.neighbors import BallTree

# Our own imports
from . import _simulation
from .consts import (
    NUM_RED, NUM_GREEN, AGGREGATE_RADIUS, SAME_CELL_RADIUS, NEIGHBOR_RADIUS,
)

# Neighbor counting


def nearest_neighbors(red_points: np.ndarray,
                      green_points: np.ndarray,
                      num_closest: int = 1) -> np.ndarray:
    """ Find the closest green point to a red point

    :param red_points:
        The n x 3 array of red points
    :param green_points:
        The m x 3 array of green points
    :param num_closest:
        The nth closest point to return
    :returns:
        An n x 3 array of distances to green points for each red point
    """

    red_points = np.stack(red_points, axis=1)
    green_points = np.stack(green_points, axis=1)

    tree = BallTree(green_points)
    return tree.query(red_points, k=num_closest, return_distance=True)[0][:, num_closest-1]


def count_neighbors(red_points: np.ndarray,
                    green_points: np.ndarray,
                    radius: float = NEIGHBOR_RADIUS) -> np.ndarray:
    """ Count the number of neighbors within a radius

    :param ndarray red_points:
        The n x 3 array of red points
    :param ndarray green_points:
        The m x 3 array of green points
    :param float radius:
        The radius within which a point is a neighbor
    :returns:
        An n x 3 array of counts of green points near each red point
    """

    red_points = np.stack(red_points, axis=1)
    green_points = np.stack(green_points, axis=1)

    tree = BallTree(green_points)
    return tree.query_radius(red_points, r=radius, count_only=True)


# Point manipulation tools


def mask_points(points: List[np.ndarray],
                mask: np.ndarray) -> Tuple[np.ndarray]:
    """ Mask off the points

    :param List[ndarray] points:
        List of 1D point arrays to mask
    :param ndarray mask:
        Mask for those arrays
    :returns:
        The same set of points, but masked
    """
    points = np.stack(points, axis=1)
    points = points[mask, :]
    return points[:, 0], points[:, 1], points[:, 2]


def concat_points(*args) -> Tuple[np.ndarray]:
    """ Concatenate all the points

    :param \\*args:
        List of ndarray tuples to concatenate
    :returns:
        An x, y, z tuple of all the points
    """

    final_x = []
    final_y = []
    final_z = []

    for (x, y, z) in args:
        if x.ndim == 0:
            assert y.ndim == 0
            assert z.ndim == 0
            continue
        assert x.shape[0] == y.shape[0]
        assert x.shape[0] == z.shape[0]
        final_x.append(x)
        final_y.append(y)
        final_z.append(z)

    return (np.concatenate(final_x),
            np.concatenate(final_y),
            np.concatenate(final_z))


def split_red_green(all_points: Tuple[np.ndarray],
                    num_red: int = NUM_RED,
                    num_green: int = NUM_GREEN,
                    udist: str = 'uniform') -> Tuple[Tuple[np.ndarray]]:
    """ Split into red and green cells

    :param Tuple[ndarray] all_points:
        The list of coordinates to split into red and green
    :param int num_red:
        The number of points to assign to red
    :param int num_green:
        The number of points to assign to green
    :param str udist:
        Distribution for the red points
    :returns:
        A tuple of (red, green) points
    """

    x, y, z = all_points
    all_radii = np.sqrt(x**2 + y**2 + z**2)
    all_indices = np.arange(all_radii.shape[0])

    # Various distributions
    if udist == 'uniform':
        all_prob = np.ones_like(all_radii)
        all_prob = all_prob / np.sum(all_prob)
    elif udist == 'left_triangle':
        all_prob = np.max(all_radii) - all_radii
        all_prob = all_prob / np.sum(all_prob)
    elif udist == 'right_triangle':
        all_prob = all_radii / np.sum(all_radii)
    elif udist == 'inside':
        sorted_indexes = np.argsort(all_radii)
        all_mask = np.zeros_like(all_radii)
        all_mask[sorted_indexes[:num_red]] = 1
        all_prob = all_mask / np.sum(all_mask)
    elif udist == 'outside':
        sorted_indexes = np.argsort(all_radii)
        all_mask = np.zeros_like(all_radii)
        all_mask[sorted_indexes[-num_red:]] = 1
        all_prob = all_mask / np.sum(all_mask)
    else:
        raise ValueError(f'Unknown distribution: {udist}')

    # Choose red cells with the probability given by the distribution
    red_indices = np.random.choice(all_indices, size=(num_red, ), p=all_prob, replace=False)

    # Now choose green cells as the remainder
    green_mask = np.ones_like(all_prob, dtype=np.bool)
    green_mask[red_indices] = False
    green_indices = all_indices[green_mask]

    # Split the coordinate masks
    red_points = (x[red_indices], y[red_indices], z[red_indices])
    green_points = (x[green_indices], y[green_indices], z[green_indices])

    print(f'Got {red_points[0].shape[0]} red points')
    print(f'Got {green_points[0].shape[0]} green points')
    return red_points, green_points


# Shape functions


def simulate_spheres_in_sphere(num_particles: int,
                               particle_radius: float = SAME_CELL_RADIUS,
                               sphere_radius: float = AGGREGATE_RADIUS,
                               rnd=np.random,
                               umin: float = 0.0,
                               umax: float = 1.0,
                               udist: str = 'uniform') -> Tuple[np.ndarray]:
    """ Simulate a set of spheres packed in a sphere

    :param int num_points:
        The number of points to draw inside the sphere
    :param float particle_radius:
        The radius of the spherical particles to pack
    :param float sphere_radius:
        Radius of the sphere to pack into
    :param RandomState rnd:
        The random number generator
    :param float umean:
        0 for center biased, 1 for edge biased, 0.5 for no bias
    :param float urange:
        The range of the generator (1 for no bias)
    :param str udist:
        The distribution for the parameter
    :returns:
        x, y, z coordinates for points in the sphere
    """

    return _simulation.simulate_spheres_in_sphere(
        num_particles, particle_radius, sphere_radius, umin, umax, udist)
