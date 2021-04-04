import os, textwrap
import numpy as np
import unittest
from context import dpti

class TestEquiThermoSetting(unittest.TestCase):
    def setUp(self) :
        self.maxDiff = None

    def test_gen_equi_thermo_settings(self):
        input = dict(timestep=0.002)
        ret1 = textwrap.dedent("""\
        # --------------------- MD SETTINGS ----------------------
        neighbor        1.0 bin
        timestep        0.002000
        thermo          ${THERMO_FREQ}
        compute         allmsd all msd
        thermo_style    custom step ke pe etotal enthalpy temp press vol lx ly lz xy xz yz pxx pyy pzz pxy pxz pyz c_allmsd[*]
        """)
        ret2 = dpti.equi.gen_equi_thermo_settings(**input)
        self.assertEqual(ret1, ret2)

if __name__ == '__main__':
    unittest.main()
