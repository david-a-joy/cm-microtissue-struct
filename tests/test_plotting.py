#!/usr/bin/env python3

# Imports
import unittest

# 3rd party
import numpy as np

# Our own imports
from cm_microtissue_struct import plotting

# Tests


class TestBootstrapCI(unittest.TestCase):

    def test_calculates_ci_correctly_1d(self):

        data = np.arange(100)

        low, high = plotting.bootstrap_ci(data, random_seed=12345)

        self.assertGreater(low, 40)
        self.assertLess(high, 60)

    def test_calculates_ci_correctly_2d(self):

        data = np.stack([
            np.arange(100),
            np.arange(100, 200),
            np.arange(200, 300),
        ], axis=1)

        low, high = plotting.bootstrap_ci(data, random_seed=12345)

        np.testing.assert_array_less(high, np.array([60, 160, 260]))
        np.testing.assert_array_less(np.array([40, 140, 240]), low)


class TestSetPlotStyle(unittest.TestCase):

    def test_can_get_current_plot_style(self):

        res = plotting.set_plot_style.get_active_style()
        self.assertIsNone(res)

        with plotting.set_plot_style('light'):
            res = plotting.set_plot_style.get_active_style()
            self.assertEqual(res, 'light')

            with plotting.set_plot_style('dark'):
                res = plotting.set_plot_style.get_active_style()
                self.assertEqual(res, 'dark')

            res = plotting.set_plot_style.get_active_style()
            self.assertEqual(res, 'light')

        res = plotting.set_plot_style.get_active_style()
        self.assertIsNone(res)
