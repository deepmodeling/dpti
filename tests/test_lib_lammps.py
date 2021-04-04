import os, textwrap
import numpy as np
import unittest
from context import dpti
# from potential_common import soft_param, soft_param_three_element, meam_model
from dpti.lib.lammps import get_natoms, get_thermo, get_last_dump
# from dpti.lib.dump import from_system_data
from numpy.testing import assert_almost_equal

class TestLibLammpsGetNatoms(unittest.TestCase):
    def setUp(self) :
        self.maxDiff = None

    def test_normal(self):
        lmp_file = os.path.join('lammps_test_files', 'test_hti.lmp')
        natoms = get_natoms(lmp_file)
        self.assertEqual(144, natoms)

    def test_raise_err(self):
        lmp_file = os.path.join('lammps_test_files', 'test_hti.lmp.broken_test_case')
        with self.assertRaises(RuntimeError):
            get_natoms(lmp_file)

class TestGetThermo(unittest.TestCase):
    def setUp(self) :
        self.maxDiff = None

        self.log_file = os.path.join('lammps_test_files', 
            'get_thermo.log.lammps')
        self.data_file = os.path.join('lammps_test_files', 
            'get_thermo.data')
        # dump fmt=%.12e

        # data1 = get_thermo(self.log_file)
        # np.savetxt(
        #     fname=self.data_file,
        #     X=data1,
        #     fmt='%.12e'
        # )

    def test_normal(self):
        data1 = get_thermo(self.log_file)
        data2 = np.loadtxt(self.data_file)
        assert_almost_equal(data1, data2, decimal=8)
    
    def test_raise_err(self):
        data1 = get_thermo(self.log_file)
        data2 = np.loadtxt(self.data_file)
        with self.assertRaises(AssertionError):
            assert_almost_equal(data1, data2, decimal=10)

class TestGetLastDump(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        self.last_dump_file = os.path.join(
            'lammps_test_files',
            'test_get_last_dump.dump.equi.lastdump'
        )
        self.dump_file = os.path.join(
            'lammps_test_files', 
            'test_get_last_dump.dump.equi'
        )

    def test_normal(self):
        with open(self.last_dump_file, 'r') as f:
            ret1_with_ending_newline = f.read()
        ret1 = ret1_with_ending_newline.strip('\n')
        ret2 = get_last_dump(self.dump_file)
        self.assertEqual(ret1, ret2)


if __name__ == '__main__':
    unittest.main()
