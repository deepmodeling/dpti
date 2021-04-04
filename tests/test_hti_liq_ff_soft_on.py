import os, textwrap
import numpy as np
import unittest
from context import dpti
from potential_common import soft_param, soft_param_three_element
from dpti.hti_liq import _ff_soft_on

class TestFfSpring(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def test_one_element(self):
        input = dict(lamb=0.075, sparam=soft_param)
        ret1 = textwrap.dedent("""\
        variable        EPSILON equal 0.030000
        pair_style      lj/cut/soft 1.000000 0.500000 6.000000
        pair_coeff      1 1 ${EPSILON} 2.493672 0.500000
        fix             tot_pot all adapt/fep 0 pair lj/cut/soft epsilon * * v_LAMBDA scale yes
        compute         e_diff all fep ${TEMP} pair lj/cut/soft epsilon * * v_EPSILON
        """)
        ret2 = _ff_soft_on(**input)
        self.assertEqual(ret1, ret2)

    def test_three_element(self):
        input = dict(lamb=0.075, sparam=soft_param_three_element)
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
        ret2 = _ff_soft_on(**input)
        self.assertEqual(ret1, ret2)
        # print(ret2)
        # pass
        # input = dict(lamb=0.075, m_spring_k=[118.71],
        #     var_spring=False)
        # ret1 = textwrap.dedent("""\
        # group           type_1 type 1
        # fix             l_spring_1 type_1 spring/self 1.1871000000e+02
        # fix_modify      l_spring_1 energy yes
        # variable        l_spring equal f_l_spring_1
        # """)
        # ret2 = dpti.hti._ff_spring(**input)
        # self.assertEqual(ret1, ret2)