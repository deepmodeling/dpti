import os, json, shutil
import numpy as np
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from context import deepti
from deepti.lib.utils import get_file_md5

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
        deepti.equi.make_task(iter_name=test_dir, jdata=jdata)
        check_file_list = ['in.lammps',  
            'conf.lmp', 'equi_settings.json', 
            'equi_cli_setting.json',
            'jdata.json', 'graph.pb']
        for file in check_file_list:
            f1 = os.path.join(benchmark_dir, file)
            f2 = os.path.join(test_dir, file)
            self.assertEqual(get_file_md5(f1), get_file_md5(f2))

    @patch('numpy.random')
    def test_npt_meam(self, patch_random):
        patch_random.randint = MagicMock(return_value=7858)
        test_name = 'npt_meam'
        benchmark_dir = os.path.join(self.benchmark_dir, test_name)
        test_dir = os.path.join(self.test_dir, test_name)

        json_file = os.path.join(benchmark_dir, 'jdata.json')
        with open(json_file) as f:
            jdata = json.load(f)
        deepti.equi.make_task(iter_name=test_dir, jdata=jdata)
        check_file_list = ['in.lammps', 'conf.lmp', 
            'equi_settings.json', 'jdata.json', 'equi_cli_setting.json',
            'Sn_18Metals.meam', 'library_18Metals.meam']
        for file in check_file_list:
            f1 = os.path.join(benchmark_dir, file)
            f2 = os.path.join(test_dir, file)
            self.assertEqual(get_file_md5(f1), get_file_md5(f2))

    @patch('numpy.random')
    def test_nvt(self, patch_random):
        patch_random.randint = MagicMock(return_value=7858)
        test_name = 'nvt'
        benchmark_dir = os.path.join(self.benchmark_dir, test_name)
        test_dir = os.path.join(self.test_dir, test_name)

        json_file = os.path.join(benchmark_dir, 'jdata.json')
        with open(json_file) as f:
            jdata = json.load(f)
        deepti.equi.make_task(iter_name=test_dir, jdata=jdata)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree('tmp_equi/')

        # iter_name = 'equi_test/npt'
        # equi_settings1 = self.get_correct_equi_setting1(iter_name)
        # equi_settings2 = deepti.equi.make_task(iter_name=iter_name, jdata=jdata)
        # self.assertEqual(equi_settings1, equi_settings2)
        # with open(os.path.join(iter_name, 'in.lammps'), 'r') as f:
        #     equi_lammps_input2 = f.read()
        # self.assertEqual(self.equi_lammps_input1, equi_lammps_input2)
        # link_graph_file = os.path.join(iter_name, 'graph.pb') 
        # self.assertEqual(self.graph_md5, get_file_md5(link_graph_file))

    # @patch('numpy.random')
    # def test_equi_make_task_self_consistent(self, patch_random):
    #     patch_random.randint = MagicMock(return_value=7858)
    #     iter_name = 'equi_test/npt_self_consistent'
    #     equi_settings1 = self.get_correct_equi_setting1(iter_name)
    #     equi_settings2 = deepti.equi.make_task(iter_name=iter_name, jdata=equi_settings1)
    #     self.assertEqual(equi_settings1, equi_settings2)
    #     with open(os.path.join(iter_name, 'in.lammps'), 'r') as f:
    #         equi_lammps_input2 = f.read()
    #     self.assertEqual(self.equi_lammps_input1, equi_lammps_input2)
    #     link_graph_file = os.path.join(iter_name, 'graph.pb') 
    #     self.assertEqual(self.graph_md5, get_file_md5(link_graph_file))

    # def get_correct_equi_setting1(self, iter_name):
    #     equi_settings1 = self.ori_equi_settings1.copy()
    #     correct_iter_name = os.path.join(self.correct_basename, iter_name)
    #     correct_equi_conf = os.path.join(self.correct_basename, 'conf.lmp')
    #     correct_deepmd_model = os.path.join(self.correct_basename, 'graph.pb')
    #     if equi_settings1['meam_library'] is not None:
    #         correct_meam_library = os.path.join(self.correct_basename, 'library_18Metals.meam')
    #         equi_settings1['meam_library'] = correct_meam_library
    #     if equi_settings1['meam_potential'] is not None:
    #         correct_meam_potential = os.path.join(self.correct_basename, 'Sn_18Metals.meam')
    #         equi_settings1['meam_potential'] = correct_meam_potential

    #     equi_settings1['iter_name'] = correct_iter_name
    #     equi_settings1['equi_conf'] = correct_equi_conf
    #     equi_settings1['deepmd_model'] = correct_deepmd_model
    #     return equi_settings1

    # @classmethod
    # def tearDownClass(self):
    #     shutil.rmtree('equi_test/npt')
    #     pass

if __name__ == '__main__':
    unittest.main()
