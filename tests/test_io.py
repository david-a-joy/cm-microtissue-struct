""" Tests for the IO tools """

# Imports
import textwrap

# 3rd party
import numpy as np

# Our own imports
from cm_microtissue_struct import io

from helpers import FileSystemTestCase

# Tests


class TestLoadSavePositionData(FileSystemTestCase):

    def test_reads_simple_positions(self):

        example = textwrap.dedent("""
            Position
            ====================
            Position X,Position Y,Position Z,Unit,Category,Collection,Time,ID
            161.035,101.236,70.1593,um,Spot,Position,1,0
            140.869,115.503,87.2728,um,Spot,Position,1,1
            158.437,97.2198,101.845,um,Spot,Position,1,2
            145.187,105.294,107.132,um,Spot,Position,1,3
        """).strip()

        testfile = self.tempdir / 'test.csv'
        with testfile.open('wt') as fp:
            fp.write('\n' + example + '\n')

        res = io.load_position_data(testfile)
        self.assertEqual(len(res), 3)

        exp_x = np.array([161.035, 140.869, 158.437, 145.187])
        exp_y = np.array([101.236, 115.503, 97.2198, 105.294])
        exp_z = np.array([70.1593, 87.2728, 101.845, 107.132])

        np.testing.assert_almost_equal(res[0], exp_x)
        np.testing.assert_almost_equal(res[1], exp_y)
        np.testing.assert_almost_equal(res[2], exp_z)

    def test_reads_simple_positions_weird_naming(self):

        example = textwrap.dedent("""
            Position
            ====================
            Cell Position X,Cell Position Y,Cell Position Z,Unit,Category,Collection,Time,ID
            161.035,101.236,70.1593,um,Spot,Position,1,0
            140.869,115.503,87.2728,um,Spot,Position,1,1
            158.437,97.2198,101.845,um,Spot,Position,1,2
            145.187,105.294,107.132,um,Spot,Position,1,3
        """).strip()

        testfile = self.tempdir / 'test.csv'
        with testfile.open('wt') as fp:
            fp.write('\n' + example + '\n')

        res = io.load_position_data(testfile)
        self.assertEqual(len(res), 3)

        exp_x = np.array([161.035, 140.869, 158.437, 145.187])
        exp_y = np.array([101.236, 115.503, 97.2198, 105.294])
        exp_z = np.array([70.1593, 87.2728, 101.845, 107.132])

        np.testing.assert_almost_equal(res[0], exp_x)
        np.testing.assert_almost_equal(res[1], exp_y)
        np.testing.assert_almost_equal(res[2], exp_z)

    def test_saves_simple_positions(self):

        x = np.array([161.035, 140.869, 158.437, 145.187])
        y = np.array([101.236, 115.503, 97.2198, 105.294])
        z = np.array([70.1593, 87.2728, 101.845, 107.132])

        testfile = self.tempdir / 'test.csv'

        io.save_position_data(testfile, (x, y, z))

        self.assertTrue(testfile.is_file())

        exp = textwrap.dedent("""
            Position
            ====================
            Position X,Position Y,Position Z,Unit,Category,Collection,Time,ID
            161.035,101.236,70.1593,um,Spot,Position,1,0
            140.869,115.503,87.2728,um,Spot,Position,1,1
            158.437,97.2198,101.845,um,Spot,Position,1,2
            145.187,105.294,107.132,um,Spot,Position,1,3
        """).strip()

        with testfile.open('rt') as fp:
            res = fp.read().strip()

        self.assertEqual(res, exp)

    def test_roundtrips(self):

        x = np.array([161.035, 140.869, 158.437, 145.187])
        y = np.array([101.236, 115.503, 97.2198, 105.294])
        z = np.array([70.1593, 87.2728, 101.845, 107.132])

        testfile = self.tempdir / 'test.csv'

        io.save_position_data(testfile, (x, y, z))

        res = io.load_position_data(testfile)

        np.testing.assert_almost_equal(res[0], x)
        np.testing.assert_almost_equal(res[1], y)
        np.testing.assert_almost_equal(res[2], z)
