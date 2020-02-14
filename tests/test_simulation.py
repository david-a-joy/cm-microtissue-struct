""" Tests for the IO tools """

# Imports
import unittest

# 3rd party
import numpy as np

from sklearn.neighbors import BallTree

# Our own imports
from cm_microtissue_struct import simulation

# Tests


class TestNearestNeighbors(unittest.TestCase):

    def test_calculates_distance_line_nearest(self):

        red_points = (
            np.array([-0.6, 0.0, 1.0, 2.0]),
            np.array([0.0, 0.0, 1.0, 2.0]),
            np.array([0.0, 0.0, 0.5, 0.0]),
        )
        green_points = (
            np.array([-1.0, 0.0, 1.0, 2.0]),
            np.array([0.0, 0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0, 2.0]),
        )

        dist = simulation.nearest_neighbors(red_points, green_points, num_closest=1)

        exp_dist = np.array([0.4, 0.0, 0.5, 1.73205081])
        np.testing.assert_allclose(dist, exp_dist, rtol=1e-3)

    def test_calculates_distance_line_self(self):

        red_points = (
            np.array([-0.6, 0.0, 1.0, 2.0]),
            np.array([0.0, 0.0, 1.0, 2.0]),
            np.array([0.0, 0.0, 0.5, 0.0]),
        )
        dist = simulation.nearest_neighbors(red_points, red_points, num_closest=1)

        exp_dist = np.array([0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(dist, exp_dist, rtol=1e-3)

        dist = simulation.nearest_neighbors(red_points, red_points, num_closest=2)

        exp_dist = np.array([0.6, 0.6, 1.5, 1.5])
        np.testing.assert_allclose(dist, exp_dist, rtol=1e-3)

    def test_calculates_count_radii(self):

        red_points = (
            np.array([-0.6, 0.0, 1.0, 2.0]),
            np.array([0.0, 0.0, 1.0, 2.0]),
            np.array([0.0, 0.0, 0.5, 0.0]),
        )
        green_points = (
            np.array([-1.0, 0.0, 1.0, 2.0]),
            np.array([0.0, 0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0, 2.0]),
        )

        dist = simulation.count_neighbors(red_points, green_points, radius=0)

        exp_dist = np.array([0.0, 1.0, 0.0, 0.0])
        np.testing.assert_allclose(dist, exp_dist, rtol=1e-3)

        dist = simulation.count_neighbors(red_points, green_points, radius=1)

        exp_dist = np.array([2.0, 2.0, 1.0, 0.0])
        np.testing.assert_allclose(dist, exp_dist, rtol=1e-3)

        dist = simulation.count_neighbors(red_points, green_points, radius=2)

        exp_dist = np.array([2.0, 3.0, 2.0, 1.0])
        np.testing.assert_allclose(dist, exp_dist, rtol=1e-3)


class TestSimulateSpheres(unittest.TestCase):
    """ Test the sphere simulation functions """

    def test_simulate_uniform_sphere(self):

        x, y, z = simulation.simulate_spheres_in_sphere(
            num_particles=100, particle_radius=0.5, sphere_radius=5.0)

        r = np.sqrt(x**2 + y**2 + z**2)
        self.assertEqual(r.shape[0], 100)
        self.assertTrue(np.all(r < 5.0))

        centers = np.stack([x, y, z], axis=1)

        tree = BallTree(centers)
        dist, _ = tree.query(centers, k=2)

        np.testing.assert_almost_equal(dist[:, 0], np.zeros((100, )))
        self.assertTrue(np.all(dist[:, 1] > 0.5))

    def test_simulate_uniform_shell(self):

        x, y, z = simulation.simulate_spheres_in_sphere(
            num_particles=100, particle_radius=0.5, sphere_radius=5.0, umin=0.9, umax=1.0)

        r = np.sqrt(x**2 + y**2 + z**2)
        self.assertEqual(r.shape[0], 100)
        self.assertTrue(np.all(r < 5.0))
        self.assertTrue(np.all(r > 4.0))

        centers = np.stack([x, y, z], axis=1)

        tree = BallTree(centers)
        dist, _ = tree.query(centers, k=2)

        np.testing.assert_almost_equal(dist[:, 0], np.zeros((100, )))
        self.assertTrue(np.all(dist[:, 1] > 0.5))

    def test_simulate_left_triangle_sphere(self):

        x, y, z = simulation.simulate_spheres_in_sphere(
            num_particles=100, particle_radius=0.5, sphere_radius=10.0,
            umin=0.0, umax=1.0, udist='left_triangle')

        r = np.sqrt(x**2 + y**2 + z**2)
        self.assertEqual(r.shape[0], 100)
        self.assertTrue(np.all(r < 10))

        num_inside = np.sum(r < 5)
        num_outside = np.sum(r >= 5)

        self.assertGreater(num_inside, 20)
        self.assertLess(num_outside, 80)

        centers = np.stack([x, y, z], axis=1)

        tree = BallTree(centers)
        dist, _ = tree.query(centers, k=2)

        np.testing.assert_almost_equal(dist[:, 0], np.zeros((100, )))
        self.assertTrue(np.all(dist[:, 1] > 0.5))

    def test_simulate_right_triangle_sphere(self):

        x, y, z = simulation.simulate_spheres_in_sphere(
            num_particles=100, particle_radius=0.5, sphere_radius=10.0,
            umin=0.0, umax=1.0, udist='right_triangle')

        r = np.sqrt(x**2 + y**2 + z**2)
        self.assertEqual(r.shape[0], 100)
        self.assertTrue(np.all(r < 10))

        num_inside = np.sum(r < 5)
        num_outside = np.sum(r >= 5)

        self.assertLess(num_inside, 10)
        self.assertGreater(num_outside, 90)

        centers = np.stack([x, y, z], axis=1)

        tree = BallTree(centers)
        dist, _ = tree.query(centers, k=2)

        np.testing.assert_almost_equal(dist[:, 0], np.zeros((100, )))
        self.assertTrue(np.all(dist[:, 1] > 0.5))

    def test_simulate_uniform_sphere_too_large_particles(self):

        with self.assertRaises(ValueError):
            simulation.simulate_spheres_in_sphere(
                num_particles=100, particle_radius=1.0, sphere_radius=5.0)

    def test_simulate_uniform_sphere_impossible_radii(self):

        with self.assertRaises(ValueError):
            simulation.simulate_spheres_in_sphere(
                num_particles=100, particle_radius=6.0, sphere_radius=5.0)
