import os
import shutil
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from dpti.lib.utils import (
    block_avg,
    get_template_ff_file,
    integrate_range_hti,
    normalize_template_ff_files,
    parse_seq,
    read_template_ff,
    relative_link_file,
)

lambda_seq = [
    "0.00:0.05:0.010",
    "0.05:0.15:0.020",
    "0.15:0.35:0.040",
    "0.35:1.00:0.065",
    "1",
]


class TestParseSeq(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def test_normal(self):
        array1 = parse_seq(lambda_seq)
        array2 = np.asarray(
            [
                0.0,
                0.01,
                0.02,
                0.03,
                0.04,
                0.05,
                0.07,
                0.09,
                0.11,
                0.13,
                0.15,
                0.19,
                0.23,
                0.27,
                0.31,
                0.35,
                0.415,
                0.48,
                0.545,
                0.61,
                0.675,
                0.74,
                0.805,
                0.87,
                0.935,
                1.0,
            ]
        )
        assert_almost_equal(array1, array2, decimal=10)

    def test_protect_eps(self):
        array1 = parse_seq(lambda_seq, protect_eps=1e-6)
        array2 = np.asarray(
            [
                0.000001,
                0.01,
                0.02,
                0.03,
                0.04,
                0.05,
                0.07,
                0.09,
                0.11,
                0.13,
                0.15,
                0.19,
                0.23,
                0.27,
                0.31,
                0.35,
                0.415,
                0.48,
                0.545,
                0.61,
                0.675,
                0.74,
                0.805,
                0.87,
                0.935,
                0.999999,
            ]
        )
        assert_almost_equal(array1, array2, decimal=10)

    def test_no_posi_args(self):
        with self.assertRaises(TypeError):
            parse_seq(lambda_seq, 1e-6)
            # assert_almost_equal(array1, array2, decimal=10)


class TestBlockAvg(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def test_normal(self):
        avg1 = 7.158014302
        err1 = 0.262593991
        data_file = "lammps_test_files/get_thermo.data"
        data_array = np.loadtxt(data_file)
        avg2, err2 = block_avg(data_array[:, 1])
        self.assertAlmostEqual(avg1, avg2, places=8)
        self.assertAlmostEqual(err1, err2, places=8)


class TestIntegrateRangeHti(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        # lamb

    def test_lamb_array_odd(self):
        result1 = -0.07001571298782591
        stt_err1 = 2.1394708051996743e-05
        sys_err1 = 2.0797767427780528e-07
        data = np.loadtxt("hti_test_files/odd.hti.out")
        lamb_array = data[:, 0]
        dU_array = data[:, 1]
        dU_err_array = data[:, 2]
        result2, stt_err2, sys_err2 = integrate_range_hti(
            lamb_array, dU_array, dU_err_array
        )
        self.assertAlmostEqual(result1, result2, places=8)
        self.assertAlmostEqual(stt_err1, stt_err2, places=8)
        self.assertAlmostEqual(sys_err1, sys_err2, places=8)

    def test_lamb_array_even(self):
        result1 = -35.48046669098458
        stt_err1 = 0.0001625198805022198
        sys_err1 = 8.812949063852216e-07
        data = np.loadtxt("hti_test_files/even.hti.out")
        lamb_array = data[:, 0]
        dU_array = data[:, 1]
        dU_err_array = data[:, 2]
        result2, stt_err2, sys_err2 = integrate_range_hti(
            lamb_array, dU_array, dU_err_array
        )
        self.assertAlmostEqual(result1, result2, places=8)
        self.assertAlmostEqual(stt_err1, stt_err2, places=8)
        self.assertAlmostEqual(sys_err1, sys_err2, places=8)


class TestRelativeLinkFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.mkdir("relative_link_file_test_dir/")
        pass

    def test_normal(self):
        relative_link_file("graph.pb", "relative_link_file_test_dir/")

    def test_other_place(self):
        relative_link_file("../pyproject.toml", "relative_link_file_test_dir/")

    def test_abs_path(self):
        abs_path = os.path.abspath(__file__)
        relative_link_file(abs_path, "relative_link_file_test_dir/")

    def test_abs_path_2(self):
        abs_path = os.path.abspath("../README.md")
        relative_link_file(abs_path, "relative_link_file_test_dir/")

    def test_raise_err(self):
        with self.assertRaises(RuntimeError):
            relative_link_file("../dpti/", "relative_link_file_test_dir/")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree("relative_link_file_test_dir/")


class TestTemplateForceFieldUtils(unittest.TestCase):
    def test_meam_does_not_use_default_template(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                with open("in.mlip", "w") as fp:
                    fp.write("pair_style zero 10.0\npair_coeff * *\n")
                self.assertIsNone(
                    get_template_ff_file({"model": None, "if_meam": True})
                )
                self.assertEqual(
                    get_template_ff_file(
                        {
                            "model": None,
                            "if_meam": True,
                            "template_ff": "explicit.mlip",
                        }
                    ),
                    "explicit.mlip",
                )
            finally:
                os.chdir(old_cwd)

    def test_empty_template_is_rejected(self):
        with tempfile.NamedTemporaryFile(mode="w") as fp:
            fp.write("  \n")
            fp.flush()
            with self.assertRaisesRegex(RuntimeError, "is empty"):
                read_template_ff(fp.name)

    def test_null_template_file_list_is_empty(self):
        self.assertEqual(normalize_template_ff_files({"template_ff_files": None}), [])


if __name__ == "__main__":
    unittest.main()
