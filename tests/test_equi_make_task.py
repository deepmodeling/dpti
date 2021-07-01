import os, json, shutil
import numpy as np
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from context import dpti
from dpti.lib.utils import get_file_md5
from dpti.equi import make_task

class TestEquiMakeTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.mkdir('tmp_equi/')

    def setUp(self):
        self.maxDiff = None
        self.test_dir = 'tmp_equi'
        self.benchmark_dir = 'benchmark_equi'

    @patch('numpy.random')
    def test_npt(self, patch_random):
        patch_random.randint = MagicMock(return_value=7858)
        test_name = 'npt'
        benchmark_dir = os.path.join(self.benchmark_dir, test_name)
        test_dir = os.path.join(self.test_dir, test_name)

        json_file = os.path.join(benchmark_dir, 'jdata.json')
        with open(json_file) as f:
            jdata = json.load(f)
        dpti.equi.make_task(iter_name=test_dir, jdata=jdata)
        check_file_list = ['in.lammps',  
            'conf.lmp',  'graph.pb']
        for file in check_file_list:
            f1 = os.path.join(benchmark_dir, file)
            f2 = os.path.join(test_dir, file)
            self.assertEqual(get_file_md5(f1), get_file_md5(f2), msg=(f1,f2))

    @patch('numpy.random')
    def test_npt_meam(self, patch_random):
        patch_random.randint = MagicMock(return_value=7858)
        test_name = 'npt_meam'
        benchmark_dir = os.path.join(self.benchmark_dir, test_name)
        test_dir = os.path.join(self.test_dir, test_name)

        json_file = os.path.join(benchmark_dir, 'jdata.json')
        with open(json_file) as f:
            jdata = json.load(f)
        dpti.equi.make_task(iter_name=test_dir, jdata=jdata)
        check_file_list = ['in.lammps', 'conf.lmp', 
            'Sn_18Metals.meam', 'library_18Metals.meam']
        for file in check_file_list:
            f1 = os.path.join(benchmark_dir, file)
            f2 = os.path.join(test_dir, file)
            self.assertEqual(get_file_md5(f1), get_file_md5(f2), msg=(f1,f2))

    @patch('numpy.random')
    def test_nvt(self, patch_random):
        patch_random.randint = MagicMock(return_value=7858)
        test_name = 'nvt'
        benchmark_dir = os.path.join(self.benchmark_dir, test_name)
        test_dir = os.path.join(self.test_dir, test_name)

        json_file = os.path.join(benchmark_dir, 'jdata.json')
        with open(json_file) as f:
            jdata = json.load(f)
        dpti.equi.make_task(iter_name=test_dir, jdata=jdata)
        check_file_list = ['in.lammps',  
            'conf.lmp', 'graph.pb']
        for file in check_file_list:
            f1 = os.path.join(benchmark_dir, file)
            f2 = os.path.join(test_dir, file)
            self.assertEqual(get_file_md5(f1), get_file_md5(f2), msg=(f1,f2))

    @patch('numpy.random')
    def test_nvt_use_npt_avg(self, patch_random):
        patch_random.randint = MagicMock(return_value=7858)
        test_name = 'nvt_use_npt_avg'
        benchmark_dir = os.path.join(self.benchmark_dir, test_name)
        test_dir = os.path.join(self.test_dir, test_name)

        npt_dir = os.path.join('benchmark_equi_log', 'npt')

        json_file = os.path.join(benchmark_dir, 'jdata.json')
        with open(json_file) as f:
            jdata = json.load(f)
        dpti.equi.make_task(iter_name=test_dir, jdata=jdata, npt_dir=npt_dir)

        check_file_list = ['in.lammps',  
            'npt_avg.lmp', 'graph.pb']
        for file in check_file_list:
            f1 = os.path.join(benchmark_dir, file)
            f2 = os.path.join(test_dir, file)
            self.assertEqual(get_file_md5(f1), get_file_md5(f2), msg=(f1,f2))

    @patch('numpy.random')
    def test_water_npt(self, patch_random):
        patch_random.randint = MagicMock(return_value=7858)
        test_name = 'npt_water'

        benchmark_dir = os.path.join(self.benchmark_dir, test_name)
        test_dir = os.path.join(self.test_dir, test_name)


        json_file = os.path.join(benchmark_dir, 'npt.json')
        with open(json_file) as f:
            jdata = json.load(f)
        dpti.equi.make_task(iter_name=test_dir, jdata=jdata)

        check_file_list = ['in.lammps']
        for file in check_file_list:
            f1 = os.path.join(benchmark_dir, file)
            f2 = os.path.join(test_dir, file)
            self.assertEqual(get_file_md5(f1), get_file_md5(f2), msg=(f1,f2))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree('tmp_equi/')

if __name__ == '__main__':
    unittest.main()
