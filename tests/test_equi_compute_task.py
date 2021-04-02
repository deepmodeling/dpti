import os, json, shutil
import numpy as np
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from context import deepti
from deepti.lib.utils import get_file_md5, relative_link_file
from deepti.equi import post_task

class TestEquiMakeTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.mkdir('tmp_equi_log/')

    def setUp(self):
        self.maxDiff = None
        self.test_dir = 'tmp_equi_log'
        self.benchmark_dir = 'benchmark_equi_log'

    @patch('builtins.print')
    def test_npt(self, patch_print):
        test_name = 'npt'
        benchmark_dir = os.path.join(self.benchmark_dir, test_name)
        test_dir = os.path.join(self.test_dir, test_name)
        os.mkdir(test_dir)
        link_file_list = ['log.lammps', 'equi_settings.json', 'conf.lmp']
        for file in link_file_list:
            src = os.path.join(benchmark_dir, file)
            relative_link_file(file_path=src, target_abs_dir=test_dir)

        post_task(test_dir)
        patch_print.assert_called_once()
        check_file_list = ['result', ]
        for file in check_file_list:
            f1 = os.path.join(benchmark_dir, file)
            f2 = os.path.join(test_dir, file)
            self.assertEqual(get_file_md5(f1), get_file_md5(f2))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree('tmp_equi_log/')


if __name__ == '__main__':
    unittest.main()
