import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from context import dpti


class TestTiPostTasks(unittest.TestCase):
    def test_pressure_path_does_not_add_com_energy_to_volume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            conf = os.path.join(tmpdir, "conf.lmp")
            with open(conf, "w") as fp:
                fp.write("2 atoms\n")

            task = os.path.join(tmpdir, "task.000000")
            os.mkdir(task)
            with open(os.path.join(task, "thermo.out"), "w") as fp:
                fp.write("10000")

            thermo = np.zeros((2, 12))
            thermo[:, 4] = 20.0
            thermo[:, 5] = 1600.0
            thermo[:, 6] = 10000.0
            thermo[:, 7] = 100.0

            jdata = {
                "equi_conf": conf,
                "stat_skip": 0,
                "stat_bsize": 1,
                "ens": "npt",
                "path": "p",
                "temp": 1600.0,
            }

            with patch("dpti.ti.get_thermo", return_value=thermo):
                dpti.ti.post_tasks(
                    tmpdir,
                    jdata,
                    Eo=0.0,
                    natoms=2,
                    scheme="trapezoidal",
                )

            output = np.loadtxt(os.path.join(tmpdir, "ti.out"), ndmin=2)
            volume_per_atom = output[0, 2]
            self.assertEqual(volume_per_atom, 50.0)


if __name__ == "__main__":
    unittest.main()
