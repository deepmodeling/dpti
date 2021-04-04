import os, textwrap
import numpy as np
import unittest
from context import dpti

class TestEquiThermoSetting(unittest.TestCase):
    def setUp(self) :
        self.maxDiff = None

    def test_gen_equi_dump_settings_false_ave(self):
        input = dict(if_dump_avg_posi=False)
        ret1 = """dump            1 all custom ${DUMP_FREQ} dump.equi id type x y z vx vy vz\n"""
        ret2 = dpti.equi.gen_equi_dump_settings(**input)
        self.assertEqual(ret1, ret2)

    def test_gen_equi_dump_settings_true_ave(self):
        input = dict(if_dump_avg_posi=True)
        ret1 = textwrap.dedent("""\
        compute         ru all property/atom xu yu zu
        fix             ap all ave/atom ${DUMP_FREQ} ${NREPEAT} ${NSTEPS} c_ru[1] c_ru[2] c_ru[3]
        dump            fp all custom ${NSTEPS} dump.avgposi id type f_ap[1] f_ap[2] f_ap[3]
        dump            1 all custom ${DUMP_FREQ} dump.equi id type x y z vx vy vz
        """)
        ret2 = dpti.equi.gen_equi_dump_settings(**input)
        self.assertEqual(ret1, ret2)

if __name__ == '__main__':
    unittest.main()
