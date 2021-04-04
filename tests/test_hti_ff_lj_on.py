import os, textwrap
import numpy as np
import unittest
from context import dpti
from potential_common import soft_param, soft_param_three_element
# print(dpti.equi)

class TestEquiForceField(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def test_normal(self):
        input = dict(lamb=0.075, model=None, sparam=soft_param)
        ret1 = textwrap.dedent("""\
        variable        EPSILON equal 0.030000
        pair_style      lj/cut/soft 1.000000 0.500000 6.000000
        pair_coeff      1 1 ${EPSILON} 2.493672 0.500000
        fix             tot_pot all adapt/fep 0 pair lj/cut/soft epsilon * * v_LAMBDA scale yes
        compute         e_diff all fep ${TEMP} pair lj/cut/soft epsilon * * v_EPSILON
        """)
        # ret1 = textwrap.dedent(ret1_raw)
        ret2 = dpti.hti._ff_lj_on(**input)
        # print('------')
        # print(ret1)
        # print('--------')
        # print(ret2)
        self.assertEqual(ret1, ret2)

    def test_three_element(self):
        input = dict(lamb=0.075, model=None, sparam=soft_param_three_element)
        ret1 = textwrap.dedent("""\
        variable        EPSILON equal 0.030000
        pair_style      lj/cut/soft 1.000000 0.600000 6.000000
        pair_coeff      1 1 ${EPSILON} 2.000000 0.500000
        pair_coeff      1 2 ${EPSILON} 2.010000 0.500000
        pair_coeff      1 3 ${EPSILON} 2.020000 0.500000
        pair_coeff      2 2 ${EPSILON} 2.110000 0.500000
        pair_coeff      2 3 ${EPSILON} 2.120000 0.500000
        pair_coeff      3 3 ${EPSILON} 2.220000 0.500000
        fix             tot_pot all adapt/fep 0 pair lj/cut/soft epsilon * * v_LAMBDA scale yes
        compute         e_diff all fep ${TEMP} pair lj/cut/soft epsilon * * v_EPSILON
        """)
        ret2 = dpti.hti._ff_lj_on(**input)
        self.assertEqual(ret1, ret2)

if __name__ == '__main__':
    unittest.main()
