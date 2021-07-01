#%%
import os, textwrap, json, shutil
import numpy as np
import unittest
# from potential_common import soft_param, soft_param_three_element, meam_model
# from dpti.lib.lammps import get_natoms, get_thermo, get_last_dump
# from dpti.lib.dump import from_system_data
from dpti.hti import _gen_lammps_input
from numpy.testing import assert_almost_equal
from unittest.mock import MagicMock, patch, PropertyMock

from dpti import hti_ice
from dpti.lib.utils import get_file_md5

class TestHtiIceGenLammpsInput(unittest.TestCase):
    def setUp(self) :
        self.maxDiff = None
        # with open('benchmark_hti_ice/hti_ice.json', 'r') as f:
        #     self.jdata = json.load(f)

    @patch('numpy.random')
    def test_hti_ice_tasks(self, patch_random):
        patch_random.randint = MagicMock(return_value=7858)
        args = MagicMock(output='tmp_hti_ice/new_job/', 
            switch='three-step', 
            command='gen',
            PARAM='benchmark_hti_ice/hti_ice.json'
        )
        hti_ice.exec_args(args=args, parser=None)
        check_file_list = [ 
            'conf.lmp', 
            '00.lj_on/task.000003/conf.lmp',
            '00.lj_on/task.000003/in.lammps',
            '01.deep_on/task.000004/in.lammps',
            '02.spring_off/task.000005/in.lammps'
        ]
        for file in check_file_list:
            f1 = os.path.join('benchmark_hti_ice/new_job/', file)
            f2 = os.path.join('tmp_hti_ice/new_job/', file)
            self.assertEqual(get_file_md5(f1), get_file_md5(f2), msg=(f1,f2))

    @patch('numpy.random')
    def test_hti_ice_old_json_tasks(self, patch_random):
        patch_random.randint = MagicMock(return_value=7858)
        args = MagicMock(output='tmp_hti_ice/old_json_job/', 
            switch='three-step', 
            command='gen',
            PARAM='benchmark_hti_ice/hti_ice.json.old'
        )
        hti_ice.exec_args(args=args, parser=None)
        check_file_list = [ 
            'conf.lmp', 
            '00.lj_on/task.000003/conf.lmp',
            '00.lj_on/task.000003/in.lammps',
            '01.deep_on/task.000004/in.lammps',
            '02.spring_off/task.000005/in.lammps'
        ]
        for file in check_file_list:
            f1 = os.path.join('benchmark_hti_ice/new_job/', file)
            f2 = os.path.join('tmp_hti_ice/old_json_job/', file)
            self.assertEqual(get_file_md5(f1), get_file_md5(f2), msg=(f1,f2))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree('tmp_hti_ice/')
