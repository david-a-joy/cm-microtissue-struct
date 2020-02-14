""" Polyhedron Tools

Calculate statistics on polyhedrons and nD point clouds

* :py:func:`vertices_of_polyhedron`: Find the exterior verticies of a point cloud
* :py:func:`centroid_of_polyhedron`: Find the center of mass of a convex hull
* :py:func:`volume_of_polyhedron`: Find the volume of a convex hull

"""

# 3rd party
import numpy as np

from scipy.spatial import ConvexHull

# Functions


def vertices_of_polyhedron(points: np.ndarray) -> np.ndarray:
    """ Calculate the exterior vertices of a polyhedron

    :param ndarray points:
        The n x 3 collection of points
    :returns:
        The m x 3 points on the hull
    """
    hull = ConvexHull(points)
    return points[hull.vertices, :]


def centroid_of_polyhedron(points: np.ndarray) -> np.ndarray:
    """ Calculate the controid of the convex hull

    :param ndarray points:
        The n x 3 collection of points
    :returns:
        The centroid of the hull
    """
    assert points.ndim == 2
    assert points.shape[0] > 3
    assert points.shape[1] == 3

    hull = ConvexHull(points)

    num_tris = len(hull.simplices)
    centroids = np.zeros((num_tris, points.shape[1]))
    weights = np.zeros((num_tris, ))
    for i, simplex in enumerate(hull.simplices):
        coords = points[simplex, :]
        centroids[i, :] = np.mean(coords, axis=0)

        # Heron's formula
        deltas = np.sqrt(np.sum((coords - coords[[1, 2, 0], :])**2, axis=1))
        p = np.sum(deltas) / 2
        area = np.sqrt(p*(p-deltas[0])*(p-deltas[1])*(p-deltas[2]))

        weights[i] = area
    weights = weights / np.sum(weights)
    return np.average(centroids, weights=weights, axis=0)


def volume_of_polyhedron(points: np.ndarray) -> float:
    """ Calculate the volume of a set of 3D points

    :param ndarray points:
        The n x 3 collection of points
    :returns:
        The volume of the hull
    """
    return ConvexHull(points).volume
