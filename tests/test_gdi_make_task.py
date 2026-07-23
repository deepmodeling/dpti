import json
import os
import shutil
import unittest
from unittest.mock import MagicMock, patch

from context import dpti

from dpti.lib.utils import get_file_md5


class TestGdiMakeTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.mkdir("tmp_gdi/")

    def setUp(self):
        self.maxDiff = None
        self.test_dir = "tmp_gdi"
        self.benchmark_dir = "benchmark_gdi"

    @patch("numpy.random.default_rng")
    def test_deepmd(self, patch_random):
        patch_random.return_value = MagicMock(integers=MagicMock(return_value=7858))
        test_name = "deepmd/0"
        benchmark_dir = os.path.join(self.benchmark_dir, test_name)
        test_dir = os.path.join(self.test_dir, test_name)

        json_file = os.path.join(benchmark_dir, "../", "pb.json")
        with open(json_file) as f:
            jdata = json.load(f)

        dpti.gdi._make_tasks_onephase(
            temp=300,
            pres=50000,
            task_path=test_dir,
            jdata=jdata,
            ens="npt",
            conf_file="conf.lmp",
            graph_file="graph.pb",
            if_meam=False,
            meam_model=None,
        )

        check_file_list = ["graph.pb", "conf.lmp", "in.lammps"]
        for file in check_file_list:
            f1 = os.path.join(benchmark_dir, file)
            f2 = os.path.join(test_dir, file)
            self.assertEqual(get_file_md5(f1), get_file_md5(f2), msg=(f1, f2))

    def test_setup_dpdt_phase_specific_models(self):
        parent_dir = os.path.join(self.test_dir, "phase_models")
        task_dir = os.path.join(parent_dir, "gdi_job")
        os.makedirs(parent_dir)
        shutil.copyfile("conf.lmp", os.path.join(parent_dir, "conf.0.lmp"))
        shutil.copyfile("alpha.lmp", os.path.join(parent_dir, "conf.1.lmp"))
        shutil.copyfile("graph.pb", os.path.join(parent_dir, "graph.0.pb"))
        shutil.copyfile("beta.lmp", os.path.join(parent_dir, "graph.1.pb"))
        jdata = {
            "phase_i": {
                "name": "PHASE_0",
                "equi_conf": "conf.0.lmp",
                "model": "graph.0.pb",
            },
            "phase_ii": {
                "name": "PHASE_1",
                "equi_conf": "conf.1.lmp",
                "model": "graph.1.pb",
            },
            "mass_map": [118.71],
            "nsteps": 5000,
            "timestep": 0.002,
            "tau_t": 0.1,
            "tau_p": 1.0,
            "thermo_freq": 10,
            "stat_skip": 100,
            "stat_bsize": 10,
        }

        dpti.gdi._setup_dpdt(task_dir, jdata)

        for file in ["conf.0.lmp", "conf.1.lmp", "graph.0.pb", "graph.1.pb"]:
            f1 = os.path.join(parent_dir, file)
            f2 = os.path.join(task_dir, file)
            self.assertEqual(get_file_md5(f1), get_file_md5(f2), msg=(f1, f2))

    def test_setup_dpdt_top_level_model_uses_legacy_graph_name(self):
        parent_dir = os.path.join(self.test_dir, "top_level_model")
        task_dir = os.path.join(parent_dir, "gdi_job")
        os.makedirs(parent_dir)
        shutil.copyfile("conf.lmp", os.path.join(parent_dir, "conf.0.lmp"))
        shutil.copyfile("alpha.lmp", os.path.join(parent_dir, "conf.1.lmp"))
        shutil.copyfile("graph.pb", os.path.join(parent_dir, "graph.pb"))
        jdata = {
            "phase_i": {
                "name": "PHASE_0",
                "equi_conf": "conf.0.lmp",
            },
            "phase_ii": {
                "name": "PHASE_1",
                "equi_conf": "conf.1.lmp",
            },
            "model": "graph.pb",
            "mass_map": [118.71],
            "nsteps": 5000,
            "timestep": 0.002,
            "tau_t": 0.1,
            "tau_p": 1.0,
            "thermo_freq": 10,
            "stat_skip": 100,
            "stat_bsize": 10,
        }

        dpti.gdi._setup_dpdt(task_dir, jdata)

        self.assertEqual(
            dpti.gdi._get_phase_graph_file(jdata, 0),
            "graph.pb",
        )
        self.assertEqual(
            dpti.gdi._get_phase_graph_file(jdata, 1),
            "graph.pb",
        )
        self.assertTrue(os.path.isfile(os.path.join(task_dir, "graph.pb")))
        self.assertFalse(os.path.exists(os.path.join(task_dir, "graph.0.pb")))
        self.assertFalse(os.path.exists(os.path.join(task_dir, "graph.1.pb")))

    def test_setup_dpdt_water_gdi_input_layout(self):
        parent_dir = os.path.join(self.test_dir, "water_gdi")
        gdi_dir = os.path.join(parent_dir, "gdi")
        task_dir = os.path.join(gdi_dir, "new_job")
        os.makedirs(gdi_dir)
        shutil.copyfile("graph.pb", os.path.join(parent_dir, "graph.pb"))
        shutil.copyfile("conf.lmp", os.path.join(gdi_dir, "conf.ice.150K.lmp"))
        shutil.copyfile("alpha.lmp", os.path.join(gdi_dir, "conf.water.300K.lmp"))
        jdata = {
            "model": "../graph.pb",
            "model_mass_map": [16, 1],
            "nsteps": 200000,
            "dt": 0.0005,
            "tau_t": 0.1,
            "tau_p": 0.5,
            "stat_freq": 100,
            "dump_freq": 10000,
            "stat_skip": 200,
            "stat_bsize": 100,
            "phase_i": {
                "name": "ice_Ih",
                "equi_conf": "conf.ice.150K.lmp",
                "ens": "npt-aniso",
            },
            "phase_ii": {
                "name": "water",
                "equi_conf": "conf.water.300K.lmp",
                "ens": "npt",
            },
        }

        dpti.gdi._setup_dpdt(task_dir, jdata)

        self.assertTrue(os.path.isfile(os.path.join(task_dir, "graph.pb")))
        self.assertFalse(os.path.exists(os.path.join(task_dir, "graph.0.pb")))
        self.assertFalse(os.path.exists(os.path.join(task_dir, "graph.1.pb")))
        self.assertEqual(dpti.gdi._get_phase_graph_file(jdata, 0), "graph.pb")
        self.assertEqual(dpti.gdi._get_phase_graph_file(jdata, 1), "graph.pb")

    def test_setup_dpdt_silica_phase_specific_input_layout(self):
        parent_dir = os.path.join(self.test_dir, "silica_gdi")
        task_dir = os.path.join(parent_dir, "new_job")
        os.makedirs(parent_dir)
        shutil.copyfile("conf.lmp", os.path.join(parent_dir, "conf_solid.lmp"))
        shutil.copyfile("alpha.lmp", os.path.join(parent_dir, "conf_melt.lmp"))
        shutil.copyfile("graph.pb", os.path.join(parent_dir, "graph_solid.pb"))
        shutil.copyfile("beta.lmp", os.path.join(parent_dir, "graph_melt.pb"))
        jdata = {
            "phase_i": {
                "name": "BETA_QUARTZ",
                "equi_conf": "conf_solid.lmp",
                "model": "graph_solid.pb",
                "ens": "npt-aniso",
            },
            "phase_ii": {
                "name": "MELT",
                "equi_conf": "conf_melt.lmp",
                "model": "graph_melt.pb",
                "ens": "npt-iso",
            },
            "mass_map": [28.085, 15.999],
            "nsteps": 1000000,
            "timestep": 0.002,
            "tau_t": 0.1,
            "tau_p": 1.0,
            "thermo_freq": 10,
            "dump_freq": 5000,
            "stat_skip": 25000,
            "stat_bsize": 100,
        }

        dpti.gdi._setup_dpdt(task_dir, jdata)

        for file in ["conf.0.lmp", "conf.1.lmp", "graph.0.pb", "graph.1.pb"]:
            self.assertTrue(os.path.isfile(os.path.join(task_dir, file)))
        self.assertEqual(dpti.gdi._get_phase_graph_file(jdata, 0), "graph.0.pb")
        self.assertEqual(dpti.gdi._get_phase_graph_file(jdata, 1), "graph.1.pb")

    @patch("numpy.random.default_rng")
    def test_deepmd_uses_local_graph_name(self, patch_random):
        patch_random.return_value = MagicMock(integers=MagicMock(return_value=7858))
        test_dir = os.path.join(self.test_dir, "deepmd_local_graph")
        json_file = os.path.join(self.benchmark_dir, "deepmd", "pb.json")
        with open(json_file) as f:
            jdata = json.load(f)

        dpti.gdi._make_tasks_onephase(
            temp=300,
            pres=50000,
            task_path=test_dir,
            jdata=jdata,
            ens="npt",
            conf_file="conf.lmp",
            graph_file="graph.0.pb",
            if_meam=False,
            meam_model=None,
        )

        with open(os.path.join(test_dir, "in.lammps")) as fp:
            lmp_input = fp.read()
        self.assertIn("pair_style      deepmd graph.pb", lmp_input)
        self.assertNotIn("pair_style      deepmd graph.0.pb", lmp_input)

    @patch("numpy.random.default_rng")
    def test_template_ff_onephase(self, patch_random):
        patch_random.return_value = MagicMock(integers=MagicMock(return_value=7858))
        test_dir = os.path.join(self.test_dir, "template_ff_onephase")
        template_file = os.path.join(self.test_dir, "in.mlip")
        support_file = os.path.join(self.test_dir, "input.nn")
        with open(template_file, "w") as fp:
            fp.write(
                "pair_style      hdnnp 6.3501269880 dir .\n" "pair_coeff      * * O H\n"
            )
        with open(support_file, "w") as fp:
            fp.write("n2p2 support file placeholder\n")

        dpti.gdi._make_tasks_onephase(
            temp=300,
            pres=1,
            task_path=test_dir,
            jdata={
                "mass_map": [16.0, 1.0],
                "nsteps": 1000,
                "timestep": 0.0005,
                "tau_t": 0.1,
                "tau_p": 0.5,
                "thermo_freq": 10,
            },
            ens="npt",
            conf_file="conf.lmp",
            graph_file=None,
            template_ff_file=template_file,
            template_ff_files=[support_file],
            if_meam=False,
            meam_model=None,
        )

        with open(os.path.join(test_dir, "in.lammps")) as fp:
            lmp_input = fp.read()
        self.assertIn("pair_style      hdnnp 6.3501269880 dir .", lmp_input)
        self.assertIn("pair_coeff      * * O H", lmp_input)
        self.assertNotIn("pair_style      deepmd", lmp_input)
        self.assertTrue(os.path.exists(os.path.join(test_dir, "input.nn")))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree("tmp_gdi/")


if __name__ == "__main__":
    unittest.main()
