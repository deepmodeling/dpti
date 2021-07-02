#%%
import os, textwrap, json, shutil
import numpy as np
import unittest
from context import dpti
# from potential_common import soft_param, soft_param_three_element, meam_model
# from dpti.lib.lammps import get_natoms, get_thermo, get_last_dump
# from dpti.lib.dump import from_system_data
from numpy.testing import assert_almost_equal
from unittest.mock import MagicMock, patch, PropertyMock

from dpti import hti_water
from dpti.lib.utils import get_file_md5

class TestHtiWaterGenLammpsInput(unittest.TestCase):
    def setUp(self) :
        self.maxDiff = None

    @patch('numpy.random')
    def test_hti_water_gen_tasks(self, patch_random):
        patch_random.randint = MagicMock(return_value=7858)
        args = MagicMock(output='tmp_hti_water/new_job/',  
            command='gen',
            PARAM='benchmark_hti_water/hti_water.json'
        )
        hti_water.exec_args(args=args, parser=None)
        check_file_list = [
            'conf.lmp', 
            '00.angle_on/task.000002/conf.lmp',
            '00.angle_on/task.000002/in.lammps',
            '01.deep_on/task.000003/in.lammps',
            '02.bond_angle_off/task.000004/in.lammps'
        ]
        for file in check_file_list:
            f1 = os.path.join('benchmark_hti_water/new_job/', file)
            f2 = os.path.join('tmp_hti_water/new_job/', file)
            self.assertEqual(get_file_md5(f1), get_file_md5(f2), msg=(f1,f2))

    @patch('numpy.random')
    def test_hti_water_gen_old_json_gen_tasks(self, patch_random):
        patch_random.randint = MagicMock(return_value=7858)
        args = MagicMock(output='tmp_hti_water/old_json_job/',  
            command='gen',
            PARAM='benchmark_hti_water/hti_water.json.old'
        )
        hti_water.exec_args(args=args, parser=None)
        check_file_list = [
            'conf.lmp', 
            '00.angle_on/task.000002/conf.lmp',
            '00.angle_on/task.000002/in.lammps',
            '01.deep_on/task.000003/in.lammps',
            '02.bond_angle_off/task.000004/in.lammps'
        ]
        for file in check_file_list:
            f1 = os.path.join('benchmark_hti_water/new_job/', file)
            f2 = os.path.join('tmp_hti_water/old_json_job/', file)
            self.assertEqual(get_file_md5(f1), get_file_md5(f2), msg=(f1,f2))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree('tmp_hti_water/')
