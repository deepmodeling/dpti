import unittest

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


if __name__ == "__main__":
    unittest.main()
