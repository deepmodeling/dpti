import os, json, shutil
import numpy as np
import unittest
from context import dpti
from unittest.mock import MagicMock, patch, PropertyMock
from dpti.lib.utils import get_file_md5
from dpti.gdi import _make_tasks_onephase

class TestGdiMakeTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.mkdir('tmp_gdi/')

    def setUp(self):
        self.maxDiff = None
        self.test_dir = 'tmp_gdi'
        self.benchmark_dir = 'benchmark_gdi'

    @patch('numpy.random')
    def test_deepmd(self, patch_random):
        patch_random.randint = MagicMock(return_value=7858)
        test_name = 'deepmd/0'
        benchmark_dir = os.path.join(self.benchmark_dir, test_name)
        test_dir = os.path.join(self.test_dir, test_name)

        json_file = os.path.join(benchmark_dir, '../', 'pb.json')
        with open(json_file) as f:
            jdata = json.load(f)

        dpti.gdi._make_tasks_onephase(
            temp=300,
            pres=50000,
            task_path=test_dir,
            jdata=jdata,
            ens='npt',
            conf_file='conf.lmp',
            graph_file='graph.pb',
            if_meam=False,
            meam_model=None
        )

        check_file_list = [ 
            'graph.pb',
            'conf.lmp',
            'in.lammps'
        ]
        for file in check_file_list:
            f1 = os.path.join(benchmark_dir, file)
            f2 = os.path.join(test_dir, file)
            self.assertEqual(get_file_md5(f1), get_file_md5(f2), msg=(f1,f2))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree('tmp_gdi/')

if __name__ == '__main__':
    unittest.main()
