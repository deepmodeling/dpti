import os
import numpy as np
import unittest
from context import deepti

class TestEquiThermoSetting(unittest.TestCase):
    def setUp(self) :
        self.maxDiff = None

    def test_gen_equi_thermo_settings(self):
        equi_settings = dict(timestep=0.002)
        ret1 = """# --------------------- MD SETTINGS ----------------------
neighbor        1.0 bin
timestep        0.002000
thermo          ${THERMO_FREQ}
compute         allmsd all msd
thermo_style    custom step ke pe etotal enthalpy temp press vol lx ly lz xy xz yz pxx pyy pzz pxy pxz pyz c_allmsd[*]
"""
        ret2 = deepti.equi.gen_equi_thermo_settings(equi_settings=equi_settings)
        self.assertEqual(ret1, ret2)

if __name__ == '__main__':
    unittest.main()
