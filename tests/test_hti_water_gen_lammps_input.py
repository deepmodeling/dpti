# %%
import os
import shutil
import tempfile
import unittest

# from potential_common import soft_param, soft_param_three_element, meam_model
# from dpti.lib.lammps import get_natoms, get_thermo, get_last_dump
# from dpti.lib.dump import from_system_data
from unittest.mock import MagicMock, patch

import numpy as np

from dpti import hti_water
from dpti.lib.utils import get_file_md5


class TestHtiWaterMbar(unittest.TestCase):
    def test_bond_angle_off_uses_reverse_lambda_scaling(self):
        """The disappearing bond/angle term scales with ``1 - lambda``."""
        de = np.array([2.0, -4.0])
        lambdas = np.array([0.25, 0.75])

        actual = hti_water._build_mbar_reduced_potential(de, lambdas, "bond_angle_off")

        np.testing.assert_allclose(
            actual,
            np.array([[-1.5, 3.0], [-0.5, 1.0]]),
        )


class TestHtiWaterGenLammpsInput(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    @patch("numpy.random.default_rng")
    def test_hti_water_gen_tasks(self, patch_random):
        patch_random.return_value = MagicMock(integers=MagicMock(return_value=7858))
        args = MagicMock(
            output="tmp_hti_water/new_job/",
            func=hti_water.handle_gen,
            PARAM="benchmark_hti_water/hti_water.json",
        )
        hti_water.exec_args(args=args, parser=None)
        check_file_list = [
            "conf.lmp",
            "00.angle_on/task.000002/conf.lmp",
            "00.angle_on/task.000002/in.lammps",
            "01.deep_on/task.000003/in.lammps",
            "02.bond_angle_off/task.000004/in.lammps",
        ]
        for file in check_file_list:
            f1 = os.path.join("benchmark_hti_water/new_job/", file)
            f2 = os.path.join("tmp_hti_water/new_job/", file)
            self.assertEqual(get_file_md5(f1), get_file_md5(f2), msg=(f1, f2))

    @patch("numpy.random.default_rng")
    def test_hti_water_gen_old_json_gen_tasks(self, patch_random):
        patch_random.return_value = MagicMock(integers=MagicMock(return_value=7858))
        args = MagicMock(
            output="tmp_hti_water/old_json_job/",
            func=hti_water.handle_gen,
            PARAM="benchmark_hti_water/hti_water.json.old",
        )
        hti_water.exec_args(args=args, parser=None)
        check_file_list = [
            "conf.lmp",
            "00.angle_on/task.000002/conf.lmp",
            "00.angle_on/task.000002/in.lammps",
            "01.deep_on/task.000003/in.lammps",
            "02.bond_angle_off/task.000004/in.lammps",
        ]
        for file in check_file_list:
            f1 = os.path.join("benchmark_hti_water/new_job/", file)
            f2 = os.path.join("tmp_hti_water/old_json_job/", file)
            self.assertEqual(get_file_md5(f1), get_file_md5(f2), msg=(f1, f2))

    @patch("numpy.random.default_rng")
    def test_template_ff_is_scaled_and_staged(self, patch_random):
        patch_random.return_value = MagicMock(integers=MagicMock(return_value=7858))
        with tempfile.TemporaryDirectory() as tmpdir:
            template_ff = os.path.join(tmpdir, "in.mlip")
            support_file = os.path.join(tmpdir, "input.nn")
            with open(template_ff, "w") as fp:
                fp.write("pair_style hdnnp 6.35 dir .\n" "pair_coeff * * O H\n")
            with open(support_file, "w") as fp:
                fp.write("n2p2 support file placeholder\n")

            output = os.path.join(tmpdir, "job")
            hti_water.make_tasks(
                output,
                {
                    "equi_conf": os.path.abspath("benchmark_hti_water/nvt_out.lmp"),
                    "template_ff": template_ff,
                    "template_ff_files": [support_file],
                    "lambda_angle_on": [0.5],
                    "lambda_deep_on": [0.5],
                    "lambda_bond_angle_off": [0.5],
                    "protect_eps": 1e-8,
                    "mass_map": [16.0, 1.0],
                    "bond_param": {
                        "bond_k": 4.0,
                        "bond_l": 0.9872238410688942,
                        "angle_k": 0.4,
                        "angle_t": 106.41090532956247,
                    },
                    "soft_param": {
                        "epsilon": 0.02,
                        "sigma_oo": 3.3,
                        "sigma_oh": 1.1,
                        "sigma_hh": 1.1,
                        "activation": 0.5,
                        "n": 1.0,
                        "alpha_lj": 0.5,
                        "rcut": 6.0,
                    },
                    "nsteps": 100,
                    "timestep": 0.0005,
                    "ens": "nvt",
                    "temp": 300.0,
                    "pres": 1.0,
                    "tau_t": 0.1,
                    "tau_p": 0.5,
                    "thermo_freq": 10,
                    "stat_skip": 10,
                    "stat_bsize": 10,
                },
            )

            deep_on_input = os.path.join(
                output, "01.deep_on", "task.000000", "in.lammps"
            )
            with open(deep_on_input) as fp:
                generated = fp.read()
            self.assertIn(
                "pair_style hybrid/scaled v_LAMBDA hdnnp 6.35 dir . " "1.0 lj/cut/soft",
                generated,
            )
            self.assertIn("pair_coeff * * hdnnp O H", generated)
            self.assertIn("compute         e_mlip all pair hdnnp", generated)
            self.assertIn("ebond eangle c_e_mlip", generated)
            self.assertNotIn("pair_style      deepmd", generated)
            self.assertTrue(
                os.path.isfile(
                    os.path.join(output, "01.deep_on", "task.000000", "input.nn")
                )
            )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree("tmp_hti_water/")
