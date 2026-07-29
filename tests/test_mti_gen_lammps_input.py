import json
import os
import tempfile
import unittest
from unittest.mock import patch

from dpti import mti


class TestMtiGenLammpsInput(unittest.TestCase):
    @patch.object(mti.Machine, "load_from_dict")
    @patch.object(mti, "uses_template_ff", return_value=True)
    def test_run_uses_persisted_settings(
        self, mock_uses_template_ff, mock_load_machine
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            job_dir = os.path.join(tmpdir, "job")
            os.mkdir(job_dir)
            persisted = {
                "job_type": "mass_ti",
                "template_ff": "in.mlip",
                "template_ff_files": ["input.nn"],
            }
            with open(os.path.join(job_dir, "mti_settings.json"), "w") as fp:
                json.dump(persisted, fp)
            machine_file = os.path.join(tmpdir, "machine.json")
            with open(machine_file, "w") as fp:
                json.dump({"command": "lmp", "machine": {}}, fp)

            mti.run_task(job_dir, {"job_type": "invalid"}, machine_file)

        mock_uses_template_ff.assert_called_once_with(persisted)
        mock_load_machine.assert_called_once_with({})

    def test_npt_prints_potential_energy(self):
        ret = mti._gen_lammps_input(
            conf_file="conf.lmp",
            mass_map=[16.0, 1.0],
            mass_scale=1.0,
            model="graph.pb",
            template_ff=None,
            nbeads=64,
            nsteps=1000,
            timestep=0.0005,
            ens="npt",
            temp=300,
            pres=1.0,
            tau_t=0.1,
            tau_p=0.5,
            thermo_freq=10,
            dump_freq=100,
        )

        self.assertIn(
            "thermo_style    custom step temp vol density pe f_1[5] f_1[7]",
            ret,
        )
        self.assertIn(
            '"$(step) $(temp) $(vol) $(density) $(pe) $(f_1[5]) $(f_1[7])"',
            ret,
        )
        self.assertIn("# step temp vol density pe K_prim K_cv", ret)

    def test_nvt_prints_potential_energy(self):
        ret = mti._gen_lammps_input(
            conf_file="conf.lmp",
            mass_map=[16.0, 1.0],
            mass_scale=1.0,
            model="graph.pb",
            template_ff=None,
            nbeads=64,
            nsteps=1000,
            timestep=0.0005,
            ens="nvt",
            temp=300,
            thermo_freq=10,
            dump_freq=100,
        )

        self.assertIn("thermo_style    custom step temp pe f_1[5] f_1[7]", ret)
        self.assertIn('"$(step) $(temp) $(pe) $(f_1[5]) $(f_1[7])"', ret)
        self.assertIn("# step temp pe K_prim K_cv", ret)

    def test_default_in_mlip_is_used_as_template_ff(self):
        template_ff = (
            "pair_style      hdnnp 6.3501269880 dir .\n" "pair_coeff      * * O H\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                with open("conf.lmp", "w") as fp:
                    fp.write("LAMMPS data placeholder\n")
                with open("in.mlip", "w") as fp:
                    fp.write(template_ff)

                mti.make_tasks(
                    "job",
                    {
                        "equi_conf": "conf.lmp",
                        "mass_map": [16.0, 1.0],
                        "nsteps": 1000,
                        "timestep": 0.0005,
                        "thermo_freq": 10,
                        "dump_freq": 100,
                        "ens": "npt",
                        "path": "t",
                        "temp_seq": [300],
                        "pres": 1.0,
                        "tau_t": 0.1,
                        "tau_p": 0.5,
                        "job_type": "mass_ti",
                        "mass_scale_y": [1.0],
                        "nbead": [64],
                    },
                )

                with open("job/task.000000/mass_scale_y.000000/in.lammps") as fp:
                    ret = fp.read()
            finally:
                os.chdir(cwd)

        self.assertIn(template_ff, ret)
        self.assertNotIn("pair_style      deepmd", ret)


if __name__ == "__main__":
    unittest.main()
