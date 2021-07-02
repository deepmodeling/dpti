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

from dpti import ti_water
from dpti.lib.utils import get_file_md5

class TestTiWaterGenLammpsInput(unittest.TestCase):
    def setUp(self) :
        self.maxDiff = None

    @patch('numpy.random')
    def test_ti_water_gen_tasks(self, patch_random):
        patch_random.randint = MagicMock(return_value=7858)
        args = MagicMock(
            output='tmp_ti_water/new_job/',  
            command='gen',
            PARAM='benchmark_ti_water/ti_water.json'
        )
        ti_water.exec_args(args=args, parser=None)
        check_file_list = [
            'conf.lmp', 
            'task.000003/in.lammps',
            'task.000003/graph.pb',
            'task.000003/thermo.out'
        ]
        for file in check_file_list:
            f1 = os.path.join('benchmark_ti_water/new_job/', file)
            f2 = os.path.join('tmp_ti_water/new_job/', file)
            self.assertEqual(get_file_md5(f1), get_file_md5(f2), msg=(f1,f2))

    @patch('numpy.random')
    def test_ti_water_old_json_gen_tasks(self, patch_random):
        patch_random.randint = MagicMock(return_value=7858)
        args = MagicMock(
            output='tmp_ti_water/old_json_job/',  
            command='gen',
            PARAM='benchmark_ti_water/ti_water.json.old'
        )
        ti_water.exec_args(args=args, parser=None)
        check_file_list = [
            'conf.lmp', 
            'task.000003/in.lammps',
            'task.000003/graph.pb',
            'task.000003/thermo.out'
        ]
        for file in check_file_list:
            f1 = os.path.join('benchmark_ti_water/new_job/', file)
            f2 = os.path.join('tmp_ti_water/old_json_job/', file)
            self.assertEqual(get_file_md5(f1), get_file_md5(f2), msg=(f1,f2))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree('tmp_ti_water/')
