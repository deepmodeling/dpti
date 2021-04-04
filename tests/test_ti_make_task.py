import os, json, shutil
import numpy as np
import unittest
from context import dpti
from unittest.mock import MagicMock, patch, PropertyMock
from dpti.lib.utils import get_file_md5

class TestEquiMakeTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.mkdir('tmp_ti/')

    def setUp(self):
        self.maxDiff = None
        self.test_dir = 'tmp_ti'
        self.benchmark_dir = 'benchmark_ti'

    @patch('numpy.random')
    def test_deepmd_path_t(self, patch_random):
        patch_random.randint = MagicMock(return_value=7858)
        test_name = 'path-t'
        benchmark_dir = os.path.join(self.benchmark_dir, test_name)
        test_dir = os.path.join(self.test_dir, test_name)

        json_file = os.path.join(benchmark_dir, 'jdata.json')
        with open(json_file) as f:
            jdata = json.load(f)
        # print(dpti.ti)
        # print(dir(dpti.ti))
        dpti.ti.make_tasks(iter_name=test_dir, jdata=jdata)
        check_file_list = [ 
            # 'conf.lmp', 
            # 'graph.pb',
            'task.000006/conf.lmp',
            'task.000006/graph.pb',
            'task.000006/thermo.out'
        ]
        for file in check_file_list:
            f1 = os.path.join(benchmark_dir, file)
            f2 = os.path.join(test_dir, file)
            self.assertEqual(get_file_md5(f1), get_file_md5(f2), msg=(f1,f2))


    @patch('numpy.random')
    def test_meam_path_p(self, patch_random):
        patch_random.randint = MagicMock(return_value=7858)
        test_name = 'path-p'
        benchmark_dir = os.path.join(self.benchmark_dir, test_name)
        test_dir = os.path.join(self.test_dir, test_name)

        json_file = os.path.join(benchmark_dir, 'jdata.json')
        with open(json_file) as f:
            jdata = json.load(f)
        # print(dpti.ti)
        # print(dir(dpti.ti))
        dpti.ti.make_tasks(iter_name=test_dir, jdata=jdata)
        check_file_list = [ 
            # 'conf.lmp', 
            'task.000006/conf.lmp',
            'task.000006/in.lammps',
            'task.000006/library_18Metals.meam',
            'task.000006/Sn_18Metals.meam',
            'task.000006/thermo.out'
        ]
        for file in check_file_list:
            f1 = os.path.join(benchmark_dir, file)
            f2 = os.path.join(test_dir, file)
            self.assertEqual(get_file_md5(f1), get_file_md5(f2), msg=(f1,f2))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree('tmp_ti/')

if __name__ == '__main__':
    unittest.main()
