import unittest
import os
import tempfile

from dpti import mti


class TestMtiGenLammpsInput(unittest.TestCase):
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
            "pair_style      hdnnp 6.3501269880 dir .\n"
            "pair_coeff      * * O H\n"
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

                with open(
                    "job/task.000000/mass_scale_y.000000/in.lammps"
                ) as fp:
                    ret = fp.read()
            finally:
                os.chdir(cwd)

        self.assertIn(template_ff, ret)
        self.assertNotIn("pair_style      deepmd", ret)


if __name__ == "__main__":
    unittest.main()
