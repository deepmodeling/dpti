import os, json, shutil
import numpy as np
import unittest
from context import dpti
from unittest.mock import MagicMock, patch, PropertyMock
from dpti.lib.utils import get_file_md5

class TestHtiMakeTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.mkdir('tmp_hti_liq/')

    def setUp(self):
        self.maxDiff = None
        self.test_dir = 'tmp_hti_liq'
        self.benchmark_dir = 'benchmark_hti_liq'

    @patch('numpy.random')
    def test_deepmd_three_step(self, patch_random):
        patch_random.randint = MagicMock(return_value=7858)
        test_name = 'three_step'
        benchmark_dir = os.path.join(self.benchmark_dir, test_name)
        test_dir = os.path.join(self.test_dir, test_name)

        json_file = os.path.join(benchmark_dir, 'jdata.json')
        with open(json_file) as f:
            jdata = json.load(f)
        # print(dpti.ti)
        # print(dir(dpti.ti))
        dpti.hti_liq.make_tasks(iter_name=test_dir, jdata=jdata)
        check_file_list = [ 
            'conf.lmp', 
            'graph.pb',
            '00.soft_on/task.000004/conf.lmp',
            '00.soft_on/task.000004/in.lammps',
            '01.deep_on/task.000005/graph.pb',
            '01.deep_on/task.000005/in.lammps',
            '02.soft_off/task.000006/in.lammps'
        ]
        for file in check_file_list:
            f1 = os.path.join(benchmark_dir, file)
            f2 = os.path.join(test_dir, file)
            self.assertEqual(get_file_md5(f1), get_file_md5(f2), msg=(f1,f2))

    @patch('numpy.random')
    def test_meam_three_step(self, patch_random):
        patch_random.randint = MagicMock(return_value=7858)
        test_name = 'three_step_meam'
        benchmark_dir = os.path.join(self.benchmark_dir, test_name)
        test_dir = os.path.join(self.test_dir, test_name)

        json_file = os.path.join(benchmark_dir, 'jdata.json')
        with open(json_file) as f:
            jdata = json.load(f)
        # print(dpti.ti)
        # print(dir(dpti.ti))
        dpti.hti_liq.make_tasks(iter_name=test_dir, jdata=jdata)
        check_file_list = [
            'conf.lmp', 
            '00.soft_on/task.000004/conf.lmp',
            '00.soft_on/task.000004/in.lammps',
            '01.deep_on/task.000005/Sn_18Metals.meam',
            '01.deep_on/task.000005/in.lammps',
            '02.soft_off/task.000006/library_18Metals.meam',
            '02.soft_off/task.000006/in.lammps'
        ]
        for file in check_file_list:
            f1 = os.path.join(benchmark_dir, file)
            f2 = os.path.join(test_dir, file)
            self.assertEqual(get_file_md5(f1), get_file_md5(f2), msg=(f1,f2))


    @classmethod
    def tearDownClass(cls):
        shutil.rmtree('tmp_hti_liq/')

if __name__ == '__main__':
    unittest.main()
