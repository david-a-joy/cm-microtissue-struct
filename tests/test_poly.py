#!/usr/bin/env python

# Stdlib
import unittest

# 3rd party
import numpy as np

# Our own imports
from cm_microtissue_struct import poly

# Tests


class TestPolyhedronStats(unittest.TestCase):

    def test_vertices_of_polyhedron(self):

        # Works in 2D
        points = np.array([
            [-1, -1],
            [-1, 1],
            [1, 1],
            [1, -1],
            [0, 0],
            [0.5, 0.5],
        ])
        res = poly.vertices_of_polyhedron(points)

        # Come back in clockwise order
        exp = np.array([
            [-1, 1],
            [-1, -1],
            [1, -1],
            [1, 1],
        ])
        np.testing.assert_almost_equal(res, exp)

        # Works in 3D
        points = np.array([
            [-1, -1, -1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, 1, -1],
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [1, 1, 0],
            [1, -1, 0],
            [0, 0, 0],
            [0.5, 0.5, 0.5],
        ])
        res = poly.vertices_of_polyhedron(points)

        # Comes back in input order
        exp = np.array([
            [-1, -1, -1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, 1, -1],
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, 1],
        ])
        np.testing.assert_almost_equal(res, exp)

    def test_calc_3d_centroid(self):

        # 3 x 3 x 2 cube with oversampled corner
        points = np.array([
            [0, 0, 0],
            [-1, -2, -1],
            [-1, -2, -0.9],
            [-1, -2, -0.8],
            [-1, -2, -0.7],
            [-1, -2, -0.6],
            [2, -2, -1],
            [-1, 1, -1],
            [-1, -2, 1],
            [-1, 1, 1],
            [2, -2, 1],
            [2, 1, -1],
            [2, 1, 1],
        ])
        center = poly.centroid_of_polyhedron(points)

        exp = np.array([0.5, -0.5, 0])

        np.testing.assert_almost_equal(center, exp)

    def test_calc_3d_volume(self):

        # 3 x 2 x 2 cube
        points = np.array([
            [0, 0, 0],
            [-1, -1, -1],
            [2, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [-1, 1, 1],
            [2, -1, 1],
            [2, 1, -1],
            [2, 1, 1],
        ])
        volume = poly.volume_of_polyhedron(points)

        self.assertEqual(volume, 12.0)
